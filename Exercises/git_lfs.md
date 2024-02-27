# Git LFS Tutorial

You will be using [Git Large File Storage](https://git-lfs.com/) (Git LFS) system to push your models to your submission repository.
Git LFS is an extension for GitHub to version large files.
In practice, large files such as models and datasets cannot be pushed to repositories.
Instead of storing large files directly in your Git repository, Git LFS stores a reference (or pointer) to these files in the repository, while the actual files are stored on a separate server.
This way your repository is easy to clone and pull from without needing to transfer the large files each time.
Moreover, it feels just like using Git, with similar commands.

## 1. Install Git LFS
There are 2 ways you can install Git LFS:

1. Install it using your favorite system package manager. git-lfs packages are available for Homebrew, MacPorts, dnf, and packagecloud
    - **Linux:** Debian and RPM packages are available from packagecloud
    - **Mac OS:** With Homebrew by doing `brew install git-lfs`
    - **Windows:** Git LFS is included in the distribution of Git for Windows. You can also install a recent version of Git LFS from the Chocolatey package manager.

2. Or download the binary and install Git LFS from [the extension's website](https://git-lfs.com/).


## 2. Git LFS Usage
1. Go to your repository's root and link git lfs:
    ```shell
    cd your-repository-path
    git lfs install
    ```

2. Create a `.gitattributes` file by tracking certain files or folders. The following commands will automatically create and append to a `.gitattributes` file:
    ```shell
    # track all large files in the models folder
    git lfs track "models/"
    # track all large files that end with ".pt"
    git lfs track "*.pt" "**.pt"
    ```

3. Commit the `.gitattributes` file first.
    ```shell
    git add .gitattributes
    git commit -m "Add attributes file"
    git push
    ```

4. Then, commit and the push the content of the folders you started tracking with Git LFS, as you would typically do with other small files:
    ```shell
    git add "models/" "*.pt" "**.pt"
    git commit -m "Add large files"
    git push
    ```

You can find more information on [the extension's website](https://git-lfs.com/) and [this Atlassian tutorial](https://www.atlassian.com/git/tutorials/git-lfs#installing-git-lfs).

## 3. Possible Problems

1. To make sure you are commiting the files to Git LFS and not to your github repository directly (which will give you an error and stop the process), do either of the following after the last `git commit`:
    - run `git lfs status` and ensure the file(s) in question appear under Git LFS objects to be committed (that they have the LFS value in parenthesis) 
    - run `git lfs ls-files` and ensure the file(s) in question appear in this output.

    More on this in [this stackoverflow post](https://stackoverflow.com/questions/54451856/how-can-i-tell-if-a-file-will-be-uploaded-to-git-lfs-correctly).

2. If you run into a problem where `git lfs status` is not listing the tracked files after the commit and before pushing, we found that depending on the system you are using, you may need to track the *complete path* to the file rather than a pattern (`"models/mymodel.pt"` instead of `"*.pt"`).