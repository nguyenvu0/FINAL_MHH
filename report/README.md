# Petri Net Analyzer - Report Compilation Guide

## 📄 Báo Cáo Đã Hoàn Thành

Báo cáo LaTeX đầy đủ theo yêu cầu đề bài CO2011 - **~14 trang** (trong giới hạn ≤15 trang).

### Structure

```
report/
├── report.tex                          # Main file
├── chapters/
│   ├── introduction.tex                # 1. Introduction (~2 trang)
│   ├── theoretical-background.tex      # 2. Theoretical Background (~3 trang)
│   ├── implementation.tex              # 3. Implementation (~4 trang)
│   ├── experimental-results.tex        # 4. Results (~3 trang)
│   └── conclusion.tex                  # 5. Conclusion (~2 trang)
└── refs/references.bib                 # 12 citations
```

---

## 🔧 Compile PDF

### Option 1: Overleaf (Khuyến Nghị - Không Cần Cài Đặt)

1. Upload thư mục `report/` lên https://www.overleaf.com/
2. Set Main Document: `report.tex`
3. Click "Recompile"

### Option 2: Local LaTeX

```bash
cd report
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

---

## ✅ Checklist Trước Khi Nộp

- [ ] Cập nhật thông tin nhóm trong `report.tex` (dòng 39-43)
- [ ] Compile và kiểm tra PDF
- [ ] Verify ≤ 15 trang
- [ ] Đóng gói ZIP theo format: `Assignment-CO2011-CSE251-{MSSV}.zip`

---

## 📚 Nội Dung Đáp Ứng Yêu Cầu

| Yêu Cầu  | Hoàn Thành |
|----------|------------|
| Theoretical background | ✅ Section 2 |
| Implementation design | ✅ Section 3 |
| Experimental results | ✅ Section 4 |
| Challenges & improvements | ✅ Section 5 |
| References | ✅ 12 citations |
