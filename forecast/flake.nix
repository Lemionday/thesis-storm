{
  description = "Micromamba + Fish Dev Shell using Nix Flakes";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux"; # or "aarch64-linux"
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          fish
          micromamba
        ];

        shell = "${pkgs.fish}/bin/fish";

        shellHook = ''
          set -gx MAMBA_ROOT_PREFIX $PWD/.mamba
          eval (micromamba shell hook --shell=fish)

          if not test -d "$MAMBA_ROOT_PREFIX/envs/dev"
            if test -f environment.yml
              micromamba create -n dev -f environment.yml -y
            else
              micromamba create -n dev python=3.11 -c conda-forge -y
            end
          end

          micromamba activate dev
          echo "✨ Welcome to your micromamba-powered Fish shell!"
        '';
      };
    };
}

