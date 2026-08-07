# Changelog

所有关于本项目的显著更改都会记录在此文件中。

本项目遵循 [Keep a Changelog](https://keepachangelog.com/) 规范，
并且采用 [Semantic Versioning](https://semver.org/) 版本控制。

## [Unreleased]

### Changed
- 调整 albumentations 版本限制

## [1.4.0] - 2026-08-01

做了一些规范化 CI/CD 和代码工程的尝试，Python 版本最低支持升到了 3.10，去除了一些不需要了的依赖。

### Added
- 新增了 dependabot。
- 引入了 ruff 0.16.0 代码检查和格式化。

### Changed
- 调整了 parser 模块的接口。

### Removed
- 去除了 render_lamply，移到 IAP 项目中。


<!-- 参考链接保持在最底部，保持正文清爽 -->
[Unreleased]: https://github.com/Lamply/CVGlue/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/Lamply/CVGlue/releases/tag/v1.4.0
