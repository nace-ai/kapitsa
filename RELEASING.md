# Releasing Kapitsa

A release is a tag `vX.Y.Z`. The workflow [.github/workflows/release.yml](.github/workflows/release.yml) then builds
the binary on a GitHub macOS runner with `packaging/release/build.sh`, attaches `kapitsa-X.Y.Z-darwin-arm64.tar.gz`
and its `.sha256` to the GitHub release (what `install.sh` downloads), installs and tests the Homebrew formula
against it, and commits the formula to nace-ai/homebrew-tap (what `brew install nace-ai/tap/kapitsa` installs). Both
install the same binary.

## Once, before the first release

1. Make this repository public (the curl installer and the release downloads need anonymous access):
   `gh repo edit nace-ai/kapitsa --visibility public --accept-visibility-change-consequences`
2. Create the tap:
   `gh repo create nace-ai/homebrew-tap --public --add-readme --description "Homebrew formulae by Nace AI"`
3. Let the workflow push to the tap: create a fine-grained token at
   <https://github.com/settings/personal-access-tokens/new> (owner nace-ai, only nace-ai/homebrew-tap, Contents read
   and write), then paste it at the prompt of
   `gh secret set TAP_GITHUB_TOKEN --repo nace-ai/kapitsa`.
   Without it, the release's job summary shows the formula to commit as `Formula/kapitsa.rb` by hand.
4. Optional, Apple signing and notarization. curl and Homebrew installs run the ad hoc signed binary as they are;
   a Developer ID signature and notarization also let a tarball downloaded in a browser pass Gatekeeper. With a
   Developer ID Application certificate exported as a .p12, and an App Store Connect API key (Users and Access >
   Integrations, role Developer):
   `base64 -i DeveloperID.p12 | gh secret set MACOS_CERTIFICATE --repo nace-ai/kapitsa`
   `gh secret set MACOS_CERTIFICATE_PASSWORD --repo nace-ai/kapitsa`
   `gh secret set NOTARY_KEY --repo nace-ai/kapitsa < AuthKey_KEYID.p8`
   `gh secret set NOTARY_KEY_ID --repo nace-ai/kapitsa --body KEYID`
   `gh secret set NOTARY_ISSUER_ID --repo nace-ai/kapitsa --body ISSUER-UUID`
   The same build signs and notarizes on a Mac with `KAPITSA_SIGN_IDENTITY="Developer ID Application: ..."` and
   `KAPITSA_NOTARY_PROFILE` (from `xcrun notarytool store-credentials`) set (see `packaging/release/build.sh`).

## Each release

1. Set the version in CMakeLists.txt (`set(KAPITSA_VERSION "X.Y.Z")`; the first release, 0.1.0, already is) and
   commit it on main: `git commit -m "Kapitsa X.Y.Z" -- CMakeLists.txt`
2. Build the release on this Mac and start a model with it once (it runs with the hardened runtime):
   `packaging/release/build.sh && ./install.sh --from-file build/release/kapitsa-X.Y.Z-darwin-arm64.tar.gz --prefix /tmp/kapitsa-check/bin && /tmp/kapitsa-check/bin/kapitsa serve`
3. Push and tag; the workflow does the rest (a tag with a `-`, like `v0.2.0-rc.1`, makes a pre-release and leaves
   the tap alone): `git push origin main && git tag vX.Y.Z && git push origin vX.Y.Z`
4. Check both installs on a Mac:
   `brew install nace-ai/tap/kapitsa && kapitsa --version`
   `curl -fsSL https://raw.githubusercontent.com/nace-ai/kapitsa/main/install.sh | bash -s -- --prefix /tmp/kapitsa-check/bin`
