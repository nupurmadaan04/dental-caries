# Contributing to Dental Caries AI (MLUA)

Thank you for your interest in contributing to the **MLUA Dental Caries Segmentation & Clinical Review Platform**!

## How to Contribute

1. **Fork the Repository** on GitHub.
2. **Clone your fork locally**:
   ```bash
   git clone https://github.com/nupurmadaan04/dental-caries.git
   cd dental-caries
   ```
3. **Create a topic branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes**:
   - For backend/engine changes: Ensure Python tests and type consistency are preserved.
   - For frontend changes: Run `npm run build` in `frontend/` to ensure zero compilation or TypeScript errors.
5. **Commit with descriptive commit messages**:
   ```bash
   git commit -m "feat: add descriptive feature explanation"
   ```
6. **Push to your branch and open a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Development Guidelines

- **Code Style:** Keep TypeScript/React components modular and formatted. Preserve dark/light theme compatibility.
- **Scientific Integrity:** Do not alter canonical checkpoint metrics (`EXP-MLUA-003_E75_BEST.pth`, $\tau = 0.50$, $71.87\%$ Val Dice; preserved historical baseline: `EXP-MLUA-003_E64_BEST.pth`, $69.39\%$ Val Dice) unless introducing a formal ablation study.
- **Code of Conduct:** Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all community interactions.
