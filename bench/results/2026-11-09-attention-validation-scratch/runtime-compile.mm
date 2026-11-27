#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>
#include <cstdlib>
int main(int argc, const char** argv) {
  if (argc != 2) return 2;
  @autoreleasepool {
    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:@(argv[1]) encoding:NSUTF8StringEncoding error:&error];
    if (!source) return 3;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 4;
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.languageVersion = (MTLLanguageVersion)(4 << 16);
    if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeFast;
    else options.fastMathEnabled = YES;
    const char* validation = getenv("MTL_SHADER_VALIDATION");
    const bool enabled = validation && atoi(validation) != 0;
    if (enabled) options.preprocessorMacros = @{@"NI_QATTN_T_VALIDATION_SCRATCH": @1};
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
      fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    printf("validation=%d Metal4 tensor library compilation PASS; no dispatch\n", enabled);
  }
  return 0;
}
