# Contributing to This Repository

This document explains how to grant others the ability to upload or push files to this repository, and how contributors can submit changes.

---

## 如何给他人上传权限 / How to Grant Others Upload Permission

### 方法一：添加协作者（直接推送权限）/ Method 1: Add a Collaborator (Direct Push Access)

If you want someone to push files directly to this repository without needing a pull request, add them as a **collaborator**:

1. Go to your repository on GitHub: `https://github.com/HorseRunningWild/MFN-for-Micro-displacement-Measurement-`
2. Click **Settings** (top menu of the repository).
3. In the left sidebar, click **Collaborators** (under "Access").
4. Click **Add people**.
5. Enter the collaborator's GitHub **username** or **email address** and click **Add [username] to this repository**.
6. The invited person will receive an email invitation. They must **accept** the invitation before they can push.

> **Permission levels you can assign:**
> | Role | Can Push | Can Manage Settings |
> |------|----------|---------------------|
> | Read | ✗ | ✗ |
> | Triage | ✗ | ✗ |
> | Write | ✓ | ✗ |
> | Maintain | ✓ | Partial |
> | Admin | ✓ | ✓ |
>
> Choose **Write** if you only want them to upload/push files.

---

### 方法二：通过 Fork 和 Pull Request 贡献 / Method 2: Fork and Pull Request (No Direct Push Access Needed)

For contributors who do **not** need direct push access, they can contribute via the standard fork-and-PR workflow:

1. **Fork** this repository by clicking the **Fork** button at the top-right of the repository page.
2. Clone their fork locally:
   ```bash
   git clone https://github.com/<their-username>/MFN-for-Micro-displacement-Measurement-.git
   cd MFN-for-Micro-displacement-Measurement-
   ```
3. Create a new branch and make changes:
   ```bash
   git checkout -b my-feature-branch
   # add or modify files
   git add .
   git commit -m "Add my changes"
   git push origin my-feature-branch
   ```
4. Open a **Pull Request** from their fork to this repository. The repository owner can then review and merge it.

---

## Code Style and File Conventions

- Python files should follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Data files (`.npy`, `.xlsx`) should match the directory structure described in [README.md](README.md#data-preparation).
- Shell scripts for SLURM jobs should follow the conventions in the existing `sbatch_*.sh` files.

---

## Questions?

If you have questions about contributing, feel free to open a [GitHub Issue](https://github.com/HorseRunningWild/MFN-for-Micro-displacement-Measurement-/issues).
