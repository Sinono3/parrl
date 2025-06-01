#!/usr/bin/env nu

let git_clean = git status --porcelain --untracked-files=no | str trim | is-empty
if not ($git_clean) {
  print "You have uncommited changes.
Please commit to proceed, as the commit hash is used for the benchmark ID."
  return;
}

let experiment = "cartpole"
let now = (date now | format date "%Y-%m-%d_%H-%M-%S")
let hash = git rev-parse --short HEAD
let path = "benchmarks" | path join $experiment $"($now)_($hash)"

let sps_path = $path | path join "benchmark.txt"
let train_path = $path | path join "train.txt"

make benchmark train
mkdir $path
./benchmark | tee { save $sps_path }
print $"SPS benchmark saved to ($sps_path)"
./train | tee { save $train_path }
print $"Train benchmark saved to ($train_path)"

