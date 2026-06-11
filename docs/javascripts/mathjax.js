window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"]
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"]
    ],
    processEscapes: true,
    processEnvironments: true
  }
  // No ignoreHtmlClass/processHtmlClass restriction: MathJax processes the
  // whole page (skipping <code>/<pre> by default). The arithmatex-only
  // restriction recommended by Material breaks mkdocs-jupyter notebooks,
  // whose prose math lives inside unclassed <p> elements that get re-ignored.
};

// Re-typeset math after Material for MkDocs instant navigation swaps the page.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
