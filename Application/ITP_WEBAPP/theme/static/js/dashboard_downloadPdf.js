// static/js/dashboard_downloadPdf.js

document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM Content Loaded for dashboard_downloadPdf.js');

    // Function to clean video title: remove common video extensions (with or without dot),
    // and normalize any existing underscores. Spaces will be preserved.
    function cleanVideoName(name) {
        if (!name) return '';
        // Remove common video extensions (case-insensitive), handling cases with and without a preceding dot
        // The (?:...) creates a non-capturing group.
        // It specifically targets the end of the string ($) to remove only actual extensions.
        let cleaned = name.replace(/(?:\.mp4|mp4|\.mov|mov|\.avi|avi|\.wmv|wmv|\.flv|flv|\.webm|webm|\.mkv|mkv)$/i, '');
        // Spaces are preserved. Only handle leading/trailing underscores and replace multiple underscores with a single one.
        cleaned = cleaned.replace(/^_|_$/g, '').replace(/_{2,}/g, '_');
        return cleaned;
    }

    // Get the main container for comparison pages, or listen globally if it's a single video page
    const comparePageContainer = document.getElementById('compare-page'); // This exists on comparison pages
    
    // Check if it's a comparison page based on the existence of comparePageContainer
    const isComparisonPage = comparePageContainer !== null;
    console.log('isComparisonPage:', isComparisonPage);

    // Attach a single event listener to a common parent that will always exist.
    // For comparison page, it's 'comparePageContainer'. For single video page, it could be 'document'.
    const eventListenerTarget = comparePageContainer || document;

    eventListenerTarget.addEventListener('click', async function (event) {
        // Check if the clicked element or its parent is the download button
        const downloadPdfBtn = event.target.closest('#download-pdf-btn');

        if (!downloadPdfBtn) {
            return; // Not the download button, do nothing
        }

        console.log('Download PDF button clicked!');

        // Dynamically find the pdfContentContainer and other elements relevant to the ACTIVE tab/page
        let pdfContentContainer;
        let tableViewDiv;
        let chartViewDiv;
        let videoTitle; // This will hold the base filename without .pdf

        if (isComparisonPage) {
            // On comparison page, find the active tab content
            const activeTabContent = document.querySelector('.tab-content:not(.hidden)');
            if (!activeTabContent) {
                console.warn('No active tab content found on comparison page.');
                return;
            }
            pdfContentContainer = activeTabContent.querySelector('#pdf-content');
            tableViewDiv = activeTabContent.querySelector('#table-view');
            chartViewDiv = activeTabContent.querySelector('#chart-view');

            const comparePageDiv = activeTabContent.closest('#compare-page'); // Get the main #compare-page to access data attributes
            if (comparePageDiv) {
                console.log('On comparison page. Getting titles from data attributes.');
                const video1Title = comparePageDiv.dataset.video1Title;
                const video2Title = comparePageDiv.dataset.video2Title;
                const proVideoTitle = comparePageDiv.dataset.videoProTitle;
                console.log('video1Title:', video1Title, 'video2Title:', video2Title, 'proVideoTitle:', proVideoTitle);

                // Determine which titles to use based on the current active tab's data-tab attribute
                const activeTabButton = document.querySelector('nav[aria-label="Comparison Tabs"] button.border-blue-600');
                let leftTitle = 'Video1';
                let rightTitle = 'Video2';

                if (activeTabButton) {
                    const currentMode = activeTabButton.dataset.tab;
                    if (currentMode === 'v1v2') {
                        leftTitle = cleanVideoName(video1Title || 'Video1');
                        rightTitle = cleanVideoName(video2Title || 'Video2');
                    } else if (currentMode === 'prov1') {
                        leftTitle = cleanVideoName(proVideoTitle || 'ProVideo');
                        rightTitle = cleanVideoName(video1Title || 'Video1');
                    } else if (currentMode === 'prov2') {
                        leftTitle = cleanVideoName(proVideoTitle || 'ProVideo');
                        rightTitle = cleanVideoName(video2Title || 'Video2');
                    }
                }
                console.log('Determined cleaned leftTitle:', leftTitle, 'cleaned rightTitle:', rightTitle);

                // --- MODIFIED: Use Video 1 & Video 2 titles directly for naming ---
                videoTitle = `${leftTitle} & ${rightTitle}`;

                console.log('Initial videoTitle for PDF filename:', videoTitle);

                const mainPdfTitle = pdfContentContainer ? pdfContentContainer.querySelector('h1') : null;
                if (mainPdfTitle) {
                    mainPdfTitle.textContent = 'Swing Comparison Analysis';
                    console.log('Main PDF H1 title set to:', mainPdfTitle.textContent);
                }

                const pdfCombinedTableTitle = pdfContentContainer ? pdfContentContainer.querySelector('h2:first-of-type') : null;
                if (pdfCombinedTableTitle) {
                    pdfCombinedTableTitle.textContent = 'Combined Table Results';
                    console.log('PDF Table H2 title set to:', pdfCombinedTableTitle.textContent);
                }
                // Update chart title for PDF
                const pdfChartTitle = pdfContentContainer ? pdfContentContainer.querySelector('#pdf-chart-title') : null;
                if (pdfChartTitle) {
                    pdfChartTitle.textContent = `${leftTitle} vs ${rightTitle} - Body Tilt Comparison`;
                    console.log('PDF Chart H2 title set to:', pdfChartTitle.textContent);
                }

            } else {
                console.warn('Could not find #compare-page div for data attributes.');
                videoTitle = 'comparison_data_default';
            }

        } else {
            // On single video results page
            pdfContentContainer = document.getElementById('pdf-content'); // Assume one exists globally for single page
            tableViewDiv = document.getElementById('table-view'); // Assume one exists globally for single page
            chartViewDiv = document.getElementById('chart-view'); // Assume one exists globally for single page

            console.log('On single video results page.');
            let rawVideoTitle = downloadPdfBtn.dataset.videoTitle || 'video_data_default';

            // --- MODIFIED: Check for combined title (Dual Upload) ---
            if (rawVideoTitle.includes(' & ')) {
                // Handle dual upload formatting: "Title1.mp4 & Title2.mp4"
                // Split, clean individual parts to remove extension, and join back.
                const parts = rawVideoTitle.split(' & ');
                const cleanedParts = parts.map(part => cleanVideoName(part.trim()));
                
                // Join them and append " Results"
                videoTitle = cleanedParts.join(' & ') + ' Results';
                
                console.log('Constructed videoTitle for dual upload:', videoTitle);
            } else {
                // Use the improved cleanVideoName directly
                let baseVideoName = cleanVideoName(rawVideoTitle);
                // Append "Results" for single upload
                videoTitle = `${baseVideoName} Results`;
                console.log('Constructed videoTitle for single page:', videoTitle);
            }

            const videoTitleElement = document.querySelector('h1.font-poppins.text-gray-800');
            if (videoTitleElement && pdfContentContainer && pdfContentContainer.querySelector('h1')) {
                pdfContentContainer.querySelector('h1').textContent = `${videoTitleElement.textContent.trim()} - Results Analysis`;
                console.log('Single video PDF H1 title set to:', pdfContentContainer.querySelector('h1').textContent);
            }
        }

        if (!pdfContentContainer) {
            console.error('PDF content container not found for the active tab/page. Cannot generate PDF.');
            return;
        }

        // Dynamically get the currently visible table and chart elements *within the active view*
        const currentVisibleTableBody = tableViewDiv ? tableViewDiv.querySelector('#combined-data-body') : null;
        const currentVisibleTiltChartCanvas = chartViewDiv ? chartViewDiv.querySelector('#tiltChart') : null;

        console.log('Found Table Element:', currentVisibleTableBody);
        console.log('Found Chart Element:', currentVisibleTiltChartCanvas);


        // Only add .pdf extension at the very end
        if (!videoTitle.toLowerCase().endsWith('.pdf')) {
            videoTitle += '.pdf';
            console.log('Added .pdf to filename. Final videoTitle:', videoTitle);
        }

        const originalPdfContentStyle = {
            position: pdfContentContainer.style.position,
            left: pdfContentContainer.style.left,
            top: pdfContentContainer.style.top,
            opacity: pdfContentContainer.style.opacity,
            pointerEvents: pdfContentContainer.style.pointerEvents,
            width: pdfContentContainer.style.width,
            height: pdfContentContainer.style.height
        };
        console.log('Storing original PDF content styles.');

        // Temporarily make pdfContentContainer visible for accurate rendering capture
        pdfContentContainer.style.position = 'static';
        pdfContentContainer.style.left = '0';
        pdfContentContainer.style.top = '0';
        pdfContentContainer.style.opacity = '1';
        pdfContentContainer.style.pointerEvents = 'auto';
        pdfContentContainer.style.width = '100%';
        pdfContentContainer.style.height = 'auto';
        console.log('PDF content container made visible for rendering.');

        // Variable to hold the chart image data URL
        let chartImageDataURL = null;

        if (isComparisonPage) {
            console.log('Processing for comparison page PDF content...');
            const pdfCombinedTableBody = pdfContentContainer.querySelector('#pdf-combined-data-body');
            const pdfCombinedTable = pdfContentContainer.querySelector('#pdf-combined-table');
            const pdfChartTitle = pdfContentContainer.querySelector('#pdf-chart-title'); // Re-select here for clarity

            if (pdfCombinedTableBody) {
                pdfCombinedTableBody.innerHTML = '';
                console.log('Cleared pdfCombinedTableBody.');
            }

            // Use the dynamically fetched visible table body
            if (currentVisibleTableBody && pdfCombinedTableBody) {
                console.log('Populating PDF table.');
                const visibleTableHeader = currentVisibleTableBody.closest('table').querySelector('thead');
                if (visibleTableHeader) {
                    const pdfTableHeader = pdfCombinedTableBody.parentNode.querySelector('thead');
                    if (pdfTableHeader) {
                         pdfTableHeader.innerHTML = visibleTableHeader.innerHTML;
                         pdfTableHeader.style.pageBreakInside = 'avoid'; // Ensure header doesn't break
                         pdfTableHeader.style.breakInside = 'avoid'; // For broader browser compatibility
                         console.log('Cloned table header to PDF table and set page-break-inside: avoid.');
                    } else {
                        console.warn('PDF combined table header not found.');
                    }
                }

                Array.from(currentVisibleTableBody.children).forEach(row => {
                    const newRow = row.cloneNode(true);
                    newRow.classList.remove('hover:bg-indigo-200');
                    newRow.querySelectorAll('img').forEach(img => {
                        img.style.width = '80px';
                        img.style.height = 'auto';
                    });
                    pdfCombinedTableBody.appendChild(newRow);
                });
                console.log('Cloned visible table body content to PDF table body.');

                // Apply styles to the comparison table for PDF
                if (pdfCombinedTable) {
                    pdfCombinedTable.classList.add('pdf-table-compact');
                    console.log('Applied pdf-table-compact class to comparison table.');
                }

            } else {
                console.warn('Cannot populate PDF table: currentVisibleTableBody or pdfCombinedTableBody is null. Check if the active tab content is correctly identified and contains the table.');
            }

            // --- CHART RENDERING LOGIC ---
            let originalChartViewHiddenState = false;
            if (chartViewDiv) {
                originalChartViewHiddenState = chartViewDiv.classList.contains('hidden');
                if (originalChartViewHiddenState) {
                    chartViewDiv.classList.remove('hidden');
                    console.log('Temporarily made chart-view visible to capture chart image.');
                    await new Promise(resolve => setTimeout(resolve, 50));
                }
            }

            if (currentVisibleTiltChartCanvas) {
                console.log('Attempting to get chart image data URL.');
                const currentChart = Chart.getChart(currentVisibleTiltChartCanvas);
                if (currentChart) {
                    await new Promise(resolve => {
                        const rect = currentVisibleTiltChartCanvas.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            currentVisibleTiltChartCanvas.width = rect.width;
                            currentVisibleTiltChartCanvas.height = rect.height;
                            console.log(`Setting canvas dimensions to: ${rect.width}x${rect.height}`);
                            currentChart.resize();
                            currentChart.update(); // Added this line to force redraw
                        } else {
                            console.warn('Canvas dimensions are still zero after making parent visible. Using fallback static dimensions.');
                            currentVisibleTiltChartCanvas.width = 800;
                            currentVisibleTiltChartCanvas.height = 400;
                            currentChart.resize();
                            currentChart.update(); // Added this line to force redraw
                        }

                        requestAnimationFrame(() => {
                            chartImageDataURL = currentChart.toBase64Image();
                            console.log('Chart image data URL generated.');
                            resolve();
                        });
                    });
                } else {
                    console.warn("Chart.js instance not found on currently visible tiltChartCanvas. Cannot generate chart image for PDF. Check if Chart.js is properly initialized for the active tab.");
                }
            } else {
                console.warn('Cannot generate PDF chart image: currentVisibleTiltChartCanvas is null. Check if the active tab content is correctly identified and contains the chart.');
            }

            // Restore original hidden state for chart-view div
            if (chartViewDiv && originalChartViewHiddenState) {
                chartViewDiv.classList.add('hidden');
                console.log('Restored chart-view hidden state.');
            }

            // Append the chart image to the new section container
            if (chartImageDataURL) {
                const chartImgElement = document.createElement('img');
                chartImgElement.src = chartImageDataURL;
                chartImgElement.style.width = '100%';
                chartImgElement.style.height = 'auto';
                chartImgElement.classList.add('pdf-chart-image');

                await new Promise((resolve) => {
                    chartImgElement.onload = () => {
                        console.log('Chart image loaded.');
                        resolve();
                    };
                    chartImgElement.onerror = (e) => {
                        console.error('Error loading chart image:', e);
                        resolve();
                    };
                    // Append the image directly after pdfChartTitle if it exists, otherwise to pdfContentContainer
                    if (pdfChartTitle && pdfChartTitle.parentNode === pdfContentContainer) {
                        pdfChartTitle.insertAdjacentElement('afterend', chartImgElement);
                        console.log('Chart image appended directly after pdfChartTitle.');
                    } else {
                        pdfContentContainer.appendChild(chartImgElement);
                        console.log('Chart image appended directly to pdfContentContainer.');
                    }

                    if (chartImgElement.complete && chartImgElement.naturalHeight !== 0) {
                        console.log('Chart image already complete at append time. Resolving immediately.');
                        resolve();
                    }
                });
            } else {
                console.warn('No chart image data URL available to append to PDF. Chart will not be included.');
            }

        } else {
            console.log('Processing for single video results page PDF content...');
            // In single video page, frontendHiddenCols are relevant
            const frontendHiddenCols = document.querySelectorAll('.frontend-hidden-col'); // These are global for single video page
            frontendHiddenCols.forEach(col => {
                col.style.display = 'table-cell';
                col.style.visibility = 'visible';
            });
            console.log('Frontend hidden columns made visible for single video PDF.');

            // Apply compact styles to the single video table for PDF
            const pdfDataTable = pdfContentContainer.querySelector('#pdf-data-table');
            if (pdfDataTable) {
                pdfDataTable.classList.add('pdf-table-compact');
                console.log('Applied pdf-table-compact class to single video table.');
            }

            // Add page-break-inside: avoid for table rows in the single video results page table
            const pdfDataTableRows = pdfContentContainer.querySelectorAll('#pdf-data-table tbody tr');
            pdfDataTableRows.forEach(row => {
                row.style.pageBreakInside = 'avoid';
                row.style.breakInside = 'avoid'; // For broader browser compatibility
            });
            console.log('Applied page-break-inside: avoid to table rows for single video PDF.');

            // Ensure the table header for single video page also avoids breaks
            const pdfDataTableHeader = pdfContentContainer.querySelector('#pdf-data-table thead');
            if (pdfDataTableHeader) {
                pdfDataTableHeader.style.pageBreakInside = 'avoid';
                pdfDataTableHeader.style.breakInside = 'avoid';
                console.log('Applied page-break-inside: avoid to single video table header.');
            }

            // Adjust padding for cells in the single video table
            pdfContentContainer.querySelectorAll('#pdf-data-table th, #pdf-data-table td').forEach(cell => {
                cell.style.paddingLeft = '8px'; // Reduce horizontal padding
                cell.style.paddingRight = '8px';
                cell.style.paddingTop = '4px'; // Reduce vertical padding
                cell.style.paddingBottom = '4px';
                cell.style.fontSize = '0.75rem'; // Reduce font size (text-xs in Tailwind CSS)
            });
            console.log('Adjusted cell padding and font size for single video PDF table.');
        }

        // A small delay before html2pdf to ensure DOM updates are processed
        setTimeout(() => {
            console.log('Initiating html2pdf generation...');
            html2pdf().from(pdfContentContainer).set({
                margin: [0.5, 0.5, 0.5, 0.5],
                filename: videoTitle,
                html2canvas: {
                    scale: 2,
                    useCORS: true,
                    allowTaint: true,
                    // letterRendering: true, // Might improve text clarity
                },
                jsPDF: {
                    unit: 'in',
                    format: 'letter',
                    orientation: 'landscape',
                },
                // Add pagebreak option to prevent breaking inside table rows
                pagebreak: {
                    mode: ['avoid-all', 'css', 'legacy'], // Use 'avoid-all' mode which respects 'page-break-inside'
                    avoid: ['tr', 'thead'] // Specifically avoid breaking inside table rows and table headers
                }
            }).save().then(() => {
                console.log('PDF saved successfully. Cleaning up styles.');
                Object.assign(pdfContentContainer.style, originalPdfContentStyle);

                if (isComparisonPage) {
                    const pdfCombinedTableBody = pdfContentContainer.querySelector('#pdf-combined-data-body');
                    const pdfCombinedTable = pdfContentContainer.querySelector('#pdf-combined-table');
                    if (pdfCombinedTableBody) {
                        pdfCombinedTableBody.innerHTML = '';
                        console.log('Cleared pdfCombinedTableBody after PDF generation.');
                    }
                    // Remove the dynamically added chart section container
                    const chartImgInPdf = pdfContentContainer.querySelector('.pdf-chart-image');
                    if (chartImgInPdf) {
                        chartImgInPdf.remove();
                        console.log('Removed chart image from PDF content container after PDF generation.');
                    }

                    // Restore original comparison table header
                    const pdfTableHeader = pdfContentContainer.querySelector('#pdf-combined-table thead');
                    if (pdfTableHeader) {
                        pdfTableHeader.innerHTML = `<tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Pose Class</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Body Tilt</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Video 1</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Video 2</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Difference</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Improvement Status</th>
                        </tr>`;
                        pdfTableHeader.style.pageBreakInside = ''; // Remove inline style
                        pdfTableHeader.style.breakInside = ''; // Remove inline style
                        console.log('Restored pdf-combined-table header.');
                    }

                    // Remove compact class from comparison table
                    if (pdfCombinedTable) {
                        pdfCombinedTable.classList.remove('pdf-table-compact');
                        console.log('Removed pdf-table-compact class from comparison table.');
                    }

                } else {
                    const frontendHiddenCols = document.querySelectorAll('.frontend-hidden-col');
                    frontendHiddenCols.forEach(col => {
                        col.style.display = 'none';
                        col.style.visibility = 'hidden';
                    });
                    console.log('Frontend hidden columns reverted for single video page.');

                    // Remove the dynamically added inline styles for page-break-inside
                    const pdfDataTableRows = pdfContentContainer.querySelectorAll('#pdf-data-table tbody tr');
                    pdfDataTableRows.forEach(row => {
                        row.style.pageBreakInside = '';
                        row.style.breakInside = '';
                    });
                    console.log('Removed page-break-inside styles from table rows.');

                    // Remove inline styles for header
                    const pdfDataTableHeader = pdfContentContainer.querySelector('#pdf-data-table thead');
                    if (pdfDataTableHeader) {
                        pdfDataTableHeader.style.pageBreakInside = '';
                        pdfDataTableHeader.style.breakInside = '';
                        console.log('Removed page-break-inside styles from single video table header.');
                    }

                    // Remove adjusted padding for cells in the single video table
                    pdfContentContainer.querySelectorAll('#pdf-data-table th, #pdf-data-table td').forEach(cell => {
                        cell.style.paddingLeft = ''; // Reset
                        cell.style.paddingRight = '';
                        cell.style.paddingTop = '';
                        cell.style.paddingBottom = '';
                        cell.style.fontSize = ''; // Reset
                    });
                    console.log('Reverted cell padding and font size for single video PDF table.');

                    const pdfDataTable = pdfContentContainer.querySelector('#pdf-data-table');
                    if (pdfDataTable) {
                        pdfDataTable.classList.remove('pdf-table-compact');
                    }
                }
            }).catch(error => {
                console.error('Error generating PDF:', error);
                // Ensure all cleanup is done even on error
                Object.assign(pdfContentContainer.style, originalPdfContentStyle);

                if (isComparisonPage) {
                    const pdfCombinedTableBody = pdfContentContainer.querySelector('#pdf-combined-data-body');
                    const pdfCombinedTable = pdfContentContainer.querySelector('#pdf-combined-table');
                    if (pdfCombinedTableBody) pdfCombinedTableBody.innerHTML = '';
                    const chartImgInPdf = pdfContentContainer.querySelector('.pdf-chart-image');
                    if (chartImgInPdf) {
                        chartImgInPdf.remove();
                    }
                    const pdfTableHeader = pdfContentContainer.querySelector('#pdf-combined-table thead');
                    if (pdfTableHeader) {
                        pdfTableHeader.innerHTML = `<tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Pose Class</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Body Tilt</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Video 1</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Video 2</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Difference</th>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Improvement Status</th>
                        </tr>`;
                        pdfTableHeader.style.pageBreakInside = '';
                        pdfTableHeader.style.breakInside = '';
                    }
                     if (pdfCombinedTable) {
                        pdfCombinedTable.classList.remove('pdf-table-compact');
                    }

                } else {
                    const frontendHiddenCols = document.querySelectorAll('.frontend-hidden-col');
                    frontendHiddenCols.forEach(col => {
                        col.style.display = 'none';
                        col.style.visibility = 'hidden';
                    });
                    const pdfDataTableRows = pdfContentContainer.querySelectorAll('#pdf-data-table tbody tr');
                    pdfDataTableRows.forEach(row => {
                        row.style.pageBreakInside = '';
                        row.style.breakInside = '';
                    });
                    const pdfDataTableHeader = pdfContentContainer.querySelector('#pdf-data-table thead');
                    if (pdfDataTableHeader) {
                        pdfDataTableHeader.style.pageBreakInside = '';
                        pdfDataTableHeader.style.breakInside = '';
                    }
                    pdfContentContainer.querySelectorAll('#pdf-data-table th, #pdf-data-table td').forEach(cell => {
                        cell.style.paddingLeft = '';
                        cell.style.paddingRight = '';
                        cell.style.paddingTop = '';
                        cell.style.paddingBottom = '';
                        cell.style.fontSize = '';
                    });
                    const pdfDataTable = pdfContentContainer.querySelector('#pdf-data-table');
                    if (pdfDataTable) {
                        pdfDataTable.classList.remove('pdf-table-compact');
                    }
                }
            });
        }, 300);
    });
});