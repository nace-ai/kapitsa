// HttpClient on NSURLSession (see http.h).
#import <Foundation/Foundation.h>

#include "hub/http.h"

#include <dispatch/dispatch.h>

namespace {

// One request in flight on a client (requests on one client are sequential).
struct Request {
  const std::function<bool(const uint8_t*, size_t)>* sink = nullptr;
  std::string body;
  long status = 0;
  bool cancelled = false;
  std::string host;   // the request's host: Authorization is not sent on to another host after a redirect
  std::string range;  // "bytes=a-b", kept on redirects
  bool range_ignored = false;  // a 200 to a range request: the server sends the whole file instead
  dispatch_semaphore_t done = nullptr;
  NSError* error = nil;
};

}  // namespace

@interface NIHttpDelegate : NSObject <NSURLSessionDataDelegate>
@property(nonatomic, assign) Request* req;
@end

@implementation NIHttpDelegate

- (void)URLSession:(NSURLSession*)session
              dataTask:(NSURLSessionDataTask*)task
    didReceiveResponse:(NSURLResponse*)response
     completionHandler:(void (^)(NSURLSessionResponseDisposition))handler {
  if ([response isKindOfClass:[NSHTTPURLResponse class]]) _req->status = long([(NSHTTPURLResponse*)response statusCode]);
  handler(NSURLSessionResponseAllow);
}

- (void)URLSession:(NSURLSession*)session dataTask:(NSURLSessionDataTask*)task didReceiveData:(NSData*)data {
  Request* r = _req;
  if (r->cancelled || r->range_ignored) return;
  if (!r->range.empty() && r->status == 200) {  // never read a whole multi-GB file into memory or the wrong offset
    r->range_ignored = true;
    [task cancel];
    return;
  }
  const bool success = r->status >= 200 && r->status < 300;
  if (!success || !r->sink) {
    [data enumerateByteRangesUsingBlock:^(const void* bytes, NSRange range, BOOL* stop) {
      if (!success && r->body.size() >= 4096) {
        *stop = YES;
        return;
      }
      r->body.append(static_cast<const char*>(bytes), range.length);
    }];
    return;
  }
  __block bool stop_all = false;
  [data enumerateByteRangesUsingBlock:^(const void* bytes, NSRange range, BOOL* stop) {
    if (!(*r->sink)(static_cast<const uint8_t*>(bytes), range.length)) {
      stop_all = true;
      *stop = YES;
    }
  }];
  if (stop_all) {
    r->cancelled = true;
    [task cancel];
  }
}

- (void)URLSession:(NSURLSession*)session
                          task:(NSURLSessionTask*)task
    willPerformHTTPRedirection:(NSHTTPURLResponse*)response
                    newRequest:(NSURLRequest*)request
             completionHandler:(void (^)(NSURLRequest*))handler {
  NSMutableURLRequest* m = [request mutableCopy];
  const char* host = request.URL.host.UTF8String;
  if (!host || _req->host != host) [m setValue:nil forHTTPHeaderField:@"Authorization"];
  if (!_req->range.empty()) [m setValue:@(_req->range.c_str()) forHTTPHeaderField:@"Range"];
  handler(m);
}

- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task didCompleteWithError:(NSError*)error {
  _req->error = error;
  dispatch_semaphore_signal(_req->done);
}

@end

namespace ni::hub {

struct HttpClient::Impl {
  NSURLSession* session = nil;
  NIHttpDelegate* delegate = nil;
};

HttpClient::HttpClient(double idle_timeout_s) : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    NSURLSessionConfiguration* c = [NSURLSessionConfiguration ephemeralSessionConfiguration];
    c.timeoutIntervalForRequest = idle_timeout_s;
    c.timeoutIntervalForResource = 7 * 24 * 3600;
    c.waitsForConnectivity = NO;
    c.URLCache = nil;
    c.requestCachePolicy = NSURLRequestReloadIgnoringLocalCacheData;
    c.HTTPCookieAcceptPolicy = NSHTTPCookieAcceptPolicyNever;
    c.HTTPShouldSetCookies = NO;
    c.HTTPMaximumConnectionsPerHost = 1;
    NSOperationQueue* q = [[NSOperationQueue alloc] init];
    q.maxConcurrentOperationCount = 1;
    impl_->delegate = [[NIHttpDelegate alloc] init];
    impl_->session = [NSURLSession sessionWithConfiguration:c delegate:impl_->delegate delegateQueue:q];
  }
}

HttpClient::~HttpClient() {
  @autoreleasepool {
    [impl_->session invalidateAndCancel];  // releases the delegate
    impl_->session = nil;
  }
}

static HttpResult run(NSURLSession* session, NIHttpDelegate* delegate, const std::string& url, const Headers& headers,
                      uint64_t a, uint64_t b, const std::function<bool(const uint8_t*, size_t)>* sink) {
  HttpResult out;
  @autoreleasepool {
    NSURL* u = [NSURL URLWithString:@(url.c_str())];
    if (!u) {
      out.error = "bad URL " + url;
      return out;
    }
    Request r;
    r.sink = sink;
    r.host = u.host.UTF8String ? u.host.UTF8String : "";
    r.done = dispatch_semaphore_create(0);
    NSMutableURLRequest* req = [NSMutableURLRequest requestWithURL:u];
    for (const auto& [k, v] : headers) [req setValue:@(v.c_str()) forHTTPHeaderField:@(k.c_str())];
    if (b > a) {
      r.range = "bytes=" + std::to_string(a) + "-" + std::to_string(b - 1);
      [req setValue:@(r.range.c_str()) forHTTPHeaderField:@"Range"];
      // byte ranges of the file as stored: no transfer compression
      [req setValue:@"identity" forHTTPHeaderField:@"Accept-Encoding"];
    }
    delegate.req = &r;
    NSURLSessionDataTask* task = [session dataTaskWithRequest:req];
    [task resume];
    dispatch_semaphore_wait(r.done, DISPATCH_TIME_FOREVER);
    delegate.req = nullptr;
    out.status = int(r.status);
    out.body = std::move(r.body);
    out.cancelled = r.cancelled;
    out.range_ignored = r.range_ignored;
    if (r.error && !r.cancelled && !r.range_ignored) {
      out.error_code = int(r.error.code);
      out.error = r.error.localizedDescription.UTF8String ? r.error.localizedDescription.UTF8String : "network error";
      if ([r.error.domain isEqualToString:NSURLErrorDomain]) {
        switch (r.error.code) {
          case NSURLErrorNotConnectedToInternet:
          case NSURLErrorCannotFindHost:
          case NSURLErrorCannotConnectToHost:
          case NSURLErrorDNSLookupFailed:
          case NSURLErrorDataNotAllowed:
          case NSURLErrorInternationalRoamingOff:
            out.offline = true;
            break;
          default:
            break;
        }
      }
    }
  }
  return out;
}

HttpResult HttpClient::get(const std::string& url, const Headers& headers, uint64_t a, uint64_t b) {
  return run(impl_->session, impl_->delegate, url, headers, a, b, nullptr);
}

HttpResult HttpClient::stream(const std::string& url, const Headers& headers, uint64_t a, uint64_t b,
                              const std::function<bool(const uint8_t*, size_t)>& sink) {
  return run(impl_->session, impl_->delegate, url, headers, a, b, &sink);
}

}  // namespace ni::hub
