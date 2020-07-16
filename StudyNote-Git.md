# git study
> MAIN SPACE
    - workspace 
    - stage
    - local branches (master as default) 
    - remote branches (origin master as default)

## Simple Process
- workspace `add` -> stage
- stage `commit` -> master
- master `push` -> origin master

## reset
- `git reset --hard <commit-id>|HEAD^|HEAD~100`
  Change to specifical commit, you can use `git log` or `git reflog` to go ahead or go to future.


## checkout
- `git checkout -- <file>` 
  Give up all changes in workspace or stage, return to last add or commit.
- `git checkout <branch-name>`
  Checkout local branch to branch-name.
- `git checkout -b <new-branch-name>`
  Create new branch and switch to it.

## proxy
`git config --global http.proxy http://hq-proxy.dyinnovations.com:8888` 
`git config --global https.proxy http://hq-proxy.dyinnovations.com:8888`
