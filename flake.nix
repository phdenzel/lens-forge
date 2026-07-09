{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };

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
