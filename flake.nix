{
  # Advertise the binary caches so local `nix develop` pulls the toolchain and
  # project closure instead of building. Nix prompts to trust these on first use.
  nixConfig = {
    extra-substituters = [
      "https://lens-forge.cachix.org"
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "lens-forge.cachix.org-1:piAThOyt1XYoV2nnyfpj2D6egL9t9CKh33g2692sHRw="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    flake-utils,
    fenix,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [fenix.overlays.default];
        };

        # Pinned Rust toolchain
        rustToolchain = pkgs.fenix.complete.withComponents [
          "cargo"
          "clippy"
          "rustc"
          "rustfmt"
          "rust-src"
          "llvm-tools-preview"
        ];

        buildInputs = with pkgs;
          [
            expat
            fontconfig
            freetype
            freetype.dev
            libGL
            libiconv
            pkg-config
          ]
          ++ lib.optionals stdenv.isLinux [
            libx11
            libxcursor
            libxi
            libxrandr
            wayland
            libxkbcommon
          ];
      in {
        devShells.default = pkgs.mkShell (
          {
            inherit buildInputs;
            packages = [rustToolchain pkgs.cargo-llvm-cov];
          }
          # LD_LIBRARY_PATH (and vulkan-loader) are Linux-only;
          # on macOS the dynamic loader/frameworks handle this.
          // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
            LD_LIBRARY_PATH =
              builtins.foldl' (a: b: "${a}:${b}/lib") "${pkgs.vulkan-loader}/lib" buildInputs;
          }
        );
      }
    );
}
