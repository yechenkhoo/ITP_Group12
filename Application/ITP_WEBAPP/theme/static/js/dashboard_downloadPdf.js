// static/js/dashboard_downloadPdf.js

document.addEventListener('DOMContentLoaded', function() {

    const downloadPdfBtn = document.getElementById('download-pdf-btn');
    console.log('DOM Content Loaded. downloadPdfBtn:', downloadPdfBtn);

    // Universal PDF content container
    const pdfContentContainer = document.getElementById('pdf-content');
    console.log('pdfContentContainer:', pdfContentContainer);

    // Elements specific to the results page (will only exist if on that page)
    const frontendHiddenCols = document.querySelectorAll('.frontend-hidden-col');
    console.log('frontendHiddenCols (for single video page):', frontendHiddenCols);

    // This check for comparison page should still be done once
    const isComparisonPage = document.getElementById('video-player-left') !== null;
    console.log('isComparisonPage:', isComparisonPage);

    // Elements within the hidden pdf-content specifically for comparison page structure
    const pdfCombinedTableBody = pdfContentContainer ? pdfContentContainer.querySelector('#pdf-combined-data-body') : null;
    const pdfTiltChartCanvas = pdfContentContainer ? pdfContentContainer.querySelector('#pdf-tiltChart') : null;
    const pdfCombinedTableTitle = pdfContentContainer ? pdfContentContainer.querySelector('#pdf-content h2:first-of-type') : null;
    const pdfChartTitle = pdfContentContainer ? pdfContentContainer.querySelector('#pdf-content h2:last-of-type') : null;
    console.log('pdfCombinedTableBody (hidden):', pdfCombinedTableBody);
    console.log('pdfTiltChartCanvas (hidden):', pdfTiltChartCanvas);
    console.log('pdfCombinedTableTitle (hidden):', pdfCombinedTableTitle);
    console.log('pdfChartTitle (hidden):', pdfChartTitle);


    if (downloadPdfBtn && pdfContentContainer) {
        downloadPdfBtn.addEventListener('click', async function () {
            console.log('Download PDF button clicked!');
            let videoTitle;
            const comparePageDiv = document.getElementById('compare-page');
            console.log('comparePageDiv:', comparePageDiv);

            // Dynamically get the currently visible table and chart elements
            let currentVisibleTableBody = null;
            let currentVisibleTiltChartCanvas = null;

            if (isComparisonPage && comparePageDiv) {
                console.log('On comparison page. Getting titles from data attributes.');
                const video1Title = comparePageDiv.dataset.video1Title;
                const video2Title = comparePageDiv.dataset.video2Title;
                const proVideoTitle = comparePageDiv.dataset.videoProTitle;
                console.log('video1Title:', video1Title, 'video2Title:', video2Title, 'proVideoTitle:', proVideoTitle);

                let leftTitle = '';
                let rightTitle = '';
                const activeTabButton = document.querySelector('nav[aria-label="Comparison Tabs"] button.border-blue-600');
                console.log('Active tab button:', activeTabButton);

                if (activeTabButton) {
                    const activeTab = activeTabButton.dataset.tab;
                    console.log('Active tab data-tab:', activeTab);
                    if (activeTab === 'v1v2') {
                        leftTitle = video1Title;
                        rightTitle = video2Title;
                    } else if (activeTab === 'prov1') {
                        leftTitle = proVideoTitle;
                        rightTitle = video1Title;
                    } else if (activeTab === 'prov2') {
                        leftTitle = proVideoTitle;
                        rightTitle = video2Title;
                    }
                }

                leftTitle = leftTitle || 'Video1';
                rightTitle = rightTitle || 'Video2';
                console.log('Determined leftTitle:', leftTitle, 'rightTitle:', rightTitle);

                // --- NEW LOGIC FOR IDENTIFYING ACTIVE TABLE AND CHART ---
                // Find the currently active (not hidden) tab content div
                const activeTabContent = document.querySelector('.tab-content:not(.hidden)');
                console.log('Active tab content div found:', activeTabContent);

                if (activeTabContent) {
                    // Query for the table body and chart canvas *within* this active tab content div
                    currentVisibleTableBody = activeTabContent.querySelector('#combined-data-body');
                    currentVisibleTiltChartCanvas = activeTabContent.querySelector('#tiltChart');
                } else {
                    console.warn('No active tab content div found. Cannot retrieve active table/chart elements.');
                }
                // --- END NEW LOGIC ---

                console.log('Found Table Element:', currentVisibleTableBody);
                console.log('Found Chart Element:', currentVisibleTiltChartCanvas);


                videoTitle = `Swing_Comparison_${leftTitle.replace(/\s+/g, '_')}_vs_${rightTitle.replace(/\s+/g, '_')}`;
                console.log('Initial videoTitle for PDF filename:', videoTitle);

                const mainPdfTitle = pdfContentContainer.querySelector('h1');
                if (mainPdfTitle) {
                    mainPdfTitle.textContent = 'Swing Comparison Analysis';
                    console.log('Main PDF H1 title set to:', mainPdfTitle.textContent);
                }

                if (pdfCombinedTableTitle) {
                    pdfCombinedTableTitle.textContent = 'Combined Table Results';
                    console.log('PDF Table H2 title set to:', pdfCombinedTableTitle.textContent);
                }
                if (pdfChartTitle) {
                    pdfChartTitle.textContent = `${leftTitle} vs ${rightTitle} - Body Tilt Comparison`;
                    console.log('PDF Chart H2 title set to:', pdfChartTitle.textContent);
                }

            } else {
                console.log('On single video results page.');
                videoTitle = downloadPdfBtn.dataset.videoTitle || 'video_data_default';
                const videoTitleElement = document.querySelector('h1.font-poppins.text-gray-800');
                if (videoTitleElement && pdfContentContainer.querySelector('h1')) {
                    pdfContentContainer.querySelector('h1').textContent = `${videoTitleElement.textContent.trim()} - Results Analysis`;
                    console.log('Single video PDF H1 title set to:', pdfContentContainer.querySelector('h1').textContent);
                }
            }

            let lowerCaseVideoTitle = videoTitle.toLowerCase();
            if (lowerCaseVideoTitle.endsWith('.mp4')) {
                videoTitle = videoTitle.substring(0, videoTitle.length - 4);
                console.log('Removed .mp4 from filename. New videoTitle:', videoTitle);
            }
            if (!videoTitle.endsWith('.pdf')) {
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

            pdfContentContainer.style.position = 'static';
            pdfContentContainer.style.left = '0';
            pdfContentContainer.style.top = '0';
            pdfContentContainer.style.opacity = '1';
            pdfContentContainer.style.pointerEvents = 'auto';
            pdfContentContainer.style.width = '100%';
            pdfContentContainer.style.height = 'auto';
            console.log('PDF content container made visible for rendering.');

            if (isComparisonPage) {
                console.log('Processing for comparison page PDF content...');
                if (pdfCombinedTableBody) {
                    pdfCombinedTableBody.innerHTML = '';
                    console.log('Cleared pdfCombinedTableBody.');
                }
                if (pdfTiltChartCanvas && Chart.getChart(pdfTiltChartCanvas)) {
                    Chart.getChart(pdfTiltChartCanvas).destroy();
                    console.log('Destroyed existing chart on pdfTiltChartCanvas.');
                }

                // Use the dynamically fetched visible table body
                if (currentVisibleTableBody && pdfCombinedTableBody) {
                    console.log('Populating PDF table.');
                    // Get the thead from the *correct* visible table
                    const visibleTableHeader = currentVisibleTableBody.closest('table').querySelector('thead');
                    if (visibleTableHeader) {
                        pdfCombinedTableBody.parentNode.querySelector('thead').innerHTML = visibleTableHeader.innerHTML;
                        console.log('Cloned table header to PDF table.');
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
                } else {
                    console.warn('Cannot populate PDF table: currentVisibleTableBody or pdfCombinedTableBody is null. Check if the active tab content is correctly identified and contains the table.');
                }

                // Use the dynamically fetched visible chart canvas
                if (currentVisibleTiltChartCanvas && pdfTiltChartCanvas) {
                    console.log('Attempting to render chart for PDF.');
                    const currentChart = Chart.getChart(currentVisibleTiltChartCanvas);
                    if (currentChart) {
                        const chartConfig = JSON.parse(JSON.stringify(currentChart.config));
                        console.log('Original chart config copied.');

                        if (chartConfig.options) {
                            chartConfig.options.animation = false;
                            chartConfig.options.responsive = true;
                            chartConfig.options.maintainAspectRatio = true;
                            if (!chartConfig.options.plugins) chartConfig.options.plugins = {};
                            if (!chartConfig.options.plugins.title) chartConfig.options.plugins.title = {};
                            chartConfig.options.plugins.title.display = true;
                            chartConfig.options.plugins.title.text = pdfChartTitle ? pdfChartTitle.textContent : `${leftTitle} vs ${rightTitle} - Body Tilt Comparison`;
                            chartConfig.options.plugins.title.font = { size: 16 };
                            console.log('Chart options modified for PDF, title set to:', chartConfig.options.plugins.title.text);
                        } else {
                            chartConfig.options = {
                                animation: false,
                                responsive: true,
                                maintainAspectRatio: true,
                                plugins: {
                                    title: {
                                        display: true,
                                        text: pdfChartTitle ? pdfChartTitle.textContent : `${leftTitle} vs ${rightTitle} - Body Tilt Comparison`,
                                        font: { size: 16 }
                                    }
                                }
                            };
                            console.log('Chart options created for PDF, title set to:', chartConfig.options.plugins.title.text);
                        }

                        await new Promise(resolve => {
                            setTimeout(() => {
                                new Chart(pdfTiltChartCanvas, chartConfig);
                                console.log('New Chart instance created on pdfTiltChartCanvas.');
                                resolve();
                            }, 300);
                        });
                    } else {
                        console.warn("Chart.js instance not found on currently visible tiltChartCanvas. Cannot render chart to PDF. Check if Chart.js is properly initialized for the active tab.");
                    }
                } else {
                    console.warn('Cannot render PDF chart: currentVisibleTiltChartCanvas or pdfTiltChartCanvas is null. Check if the active tab content is correctly identified and contains the chart.');
                }
            } else {
                console.log('Processing for single video results page PDF content...');
                frontendHiddenCols.forEach(col => {
                    col.style.display = 'table-cell';
                    col.style.visibility = 'visible';
                });
                console.log('Frontend hidden columns made visible for single video PDF.');
            }

            setTimeout(() => {
                console.log('Initiating html2pdf generation...');
                html2pdf().from(pdfContentContainer).set({
                    margin: [0.5, 0.5, 0.5, 0.5],
                    filename: videoTitle,
                    html2canvas: {
                        scale: 1,
                        useCORS: true,
                        allowTaint: true,
                    },
                    jsPDF: {
                        unit: 'in',
                        format: 'letter',
                        orientation: 'landscape',
                    }
                }).save().then(() => {
                    console.log('PDF saved successfully. Cleaning up styles.');
                    Object.assign(pdfContentContainer.style, originalPdfContentStyle);

                    if (isComparisonPage) {
                        if (pdfCombinedTableBody) {
                            pdfCombinedTableBody.innerHTML = '';
                            console.log('Cleared pdfCombinedTableBody after PDF generation.');
                        }
                        if (pdfTiltChartCanvas && Chart.getChart(pdfTiltChartCanvas)) {
                            Chart.getChart(pdfTiltChartCanvas).destroy();
                            console.log('Destroyed pdfTiltChartCanvas after PDF generation.');
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
                            console.log('Restored pdf-combined-table header.');
                        }

                    } else {
                        frontendHiddenCols.forEach(col => {
                            col.style.display = 'none';
                            col.style.visibility = 'hidden';
                        });
                        console.log('Frontend hidden columns reverted for single video page.');
                    }
                }).catch(error => {
                    console.error('Error generating PDF:', error);
                    Object.assign(pdfContentContainer.style, originalPdfContentStyle);
                    if (isComparisonPage) {
                        if (pdfCombinedTableBody) pdfCombinedTableBody.innerHTML = '';
                        if (pdfTiltChartCanvas && Chart.getChart(pdfTiltChartCanvas)) {
                            Chart.getChart(pdfTiltChartCanvas).destroy();
                        }
                    } else {
                        frontendHiddenCols.forEach(col => {
                            col.style.display = 'none';
                            col.style.visibility = 'hidden';
                        });
                    }
                });
            }, 300);
        });
    } else {
        console.warn('Download PDF button or PDF content container not found. PDF functionality disabled.');
    }
});