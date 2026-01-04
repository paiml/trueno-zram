// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">Getting Started</li><li class="chapter-item expanded "><a href="getting-started/installation.html"><strong aria-hidden="true">1.</strong> Installation</a></li><li class="chapter-item expanded "><a href="getting-started/quick-start.html"><strong aria-hidden="true">2.</strong> Quick Start</a></li><li class="chapter-item expanded "><a href="getting-started/examples.html"><strong aria-hidden="true">3.</strong> Examples</a></li><li class="chapter-item expanded affix "><li class="part-title">Core Concepts</li><li class="chapter-item expanded "><a href="concepts/algorithms.html"><strong aria-hidden="true">4.</strong> Compression Algorithms</a></li><li class="chapter-item expanded "><a href="concepts/simd.html"><strong aria-hidden="true">5.</strong> SIMD Acceleration</a></li><li class="chapter-item expanded "><a href="concepts/gpu.html"><strong aria-hidden="true">6.</strong> GPU Batch Compression</a></li><li class="chapter-item expanded "><a href="concepts/samefill.html"><strong aria-hidden="true">7.</strong> Same-Fill Detection</a></li><li class="chapter-item expanded affix "><li class="part-title">API Reference</li><li class="chapter-item expanded "><a href="api/compressor-builder.html"><strong aria-hidden="true">8.</strong> CompressorBuilder</a></li><li class="chapter-item expanded "><a href="api/gpu-batch.html"><strong aria-hidden="true">9.</strong> GPU Batch API</a></li><li class="chapter-item expanded "><a href="api/compat.html"><strong aria-hidden="true">10.</strong> Kernel Compatibility</a></li><li class="chapter-item expanded affix "><li class="part-title">Performance</li><li class="chapter-item expanded "><a href="performance/benchmarks.html"><strong aria-hidden="true">11.</strong> Benchmarks</a></li><li class="chapter-item expanded "><a href="performance/tuning.html"><strong aria-hidden="true">12.</strong> Tuning Guide</a></li><li class="chapter-item expanded "><a href="performance/pcie-rule.html"><strong aria-hidden="true">13.</strong> PCIe 5x Rule</a></li><li class="chapter-item expanded affix "><li class="part-title">Architecture</li><li class="chapter-item expanded "><a href="architecture/overview.html"><strong aria-hidden="true">14.</strong> Design Overview</a></li><li class="chapter-item expanded "><a href="architecture/simd-dispatch.html"><strong aria-hidden="true">15.</strong> SIMD Dispatch</a></li><li class="chapter-item expanded "><a href="architecture/gpu-pipeline.html"><strong aria-hidden="true">16.</strong> GPU Pipeline</a></li><li class="chapter-item expanded affix "><li class="spacer"></li><li class="chapter-item expanded affix "><a href="contributing.html">Contributing</a></li><li class="chapter-item expanded affix "><a href="changelog.html">Changelog</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
