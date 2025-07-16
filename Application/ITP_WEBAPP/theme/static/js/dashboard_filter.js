document.addEventListener('DOMContentLoaded', function () {
    const isComparePage = document.getElementById('compare-page') !== null;

    if (isComparePage) {
        // --- Compare Page Filtering Logic ---

        // Get elements for the Table Filter (now specific to compare page panels)
        const tableFilterDropdownBtn = document.getElementById('table-filter-dropdown-btn');
        const tableFilterPanel = document.getElementById('table-filter-panel');
        const tableFilterOptionsContainer = document.getElementById('table-filter-options-container');
        const tableApplyFiltersBtn = document.getElementById('table-apply-filters-btn');
        const tableClearFiltersBtn = document.getElementById('table-clear-filters-btn');

        // Get elements for the Chart Filters
        const chartTabButton = document.getElementById('chart-tab');
        const chartFiltersContainer = document.getElementById('chart-filters'); // The new div for chart checkboxes

        let tableRows = []; // Will be populated dynamically
        let tableHeaders = [];
        let tableColumnFilterDefinitions = [];
        let tableFilterStates = {}; // Stores checked state for table filters

        // Chart related variables
        let currentChartInstance = null; // Will hold the Chart.js instance
        let chartFilterStates = {}; // Stores checked state for chart datasets

        // --- Helper Functions for Compare Page Table ---

        function getCompareTableElements() {
            // This function needs to be called when the table content is rendered (e.g., after tab switch)
            const activeTableDiv = document.querySelector('#table-view:not(.hidden)');
            if (activeTableDiv) {
                const table = activeTableDiv.querySelector('table');
                if (table) {
                    tableRows = Array.from(table.querySelectorAll('tbody tr'));
                    tableHeaders = Array.from(table.querySelectorAll('thead th'));
                    return true;
                }
            }
            return false;
        }

        function prepareCompareTableFilterData() {
            if (!getCompareTableElements()) {
                console.warn("Compare Table not found or not visible. Skipping table filter data prep.");
                return;
            }

            tableColumnFilterDefinitions = [];
            const headerNameToVisibleIndexMap = new Map();

            const visibleHeaderTexts = tableHeaders.map(th => th.textContent.trim());

            visibleHeaderTexts.forEach((headerText, visibleColIndex) => {
                headerNameToVisibleIndexMap.set(headerText, visibleColIndex);
            });

            // Columns that should NOT have their own filter UI for the combined table
            const columnsToExcludeFromFilterUI = [
                'Video 1', 'Video 2', 'Difference' // These are typically numerical
            ];

            const tempFilterDefinitionsMap = new Map();

            visibleHeaderTexts.forEach((headerText, visibleColIndex) => {
                if (columnsToExcludeFromFilterUI.includes(headerText)) {
                    return;
                }

                const uniqueValues = new Set();
                tableRows.forEach(row => {
                    const cell = row.querySelectorAll('td')[visibleColIndex];
                    if (cell) {
                        uniqueValues.add(cell.textContent.trim());
                    }
                });

                let columnValues = Array.from(uniqueValues).filter(val => val !== '');

                // Special handling for Improvement Status
                if (headerText === 'Improvement Status') {
                    columnValues = ['Good', 'Bad', 'Very Bad', 'Neutral']; // Ensure full set
                } else if (headerText.trim() === 'Pose Class') { // Added .trim() for robustness
                    console.log("Applying Pose Class sort for:", columnValues);
                    columnValues.sort((a, b) => {
                        const numA = parseInt(a.substring(1)); // Extract number from "P1", "P10"
                        const numB = parseInt(b.substring(1));
                        return numA - numB; // Sort numerically
                    });
                    console.log("Sorted Pose Class values:", columnValues);
                } else {
                    columnValues.sort((a, b) => {
                        if (!isNaN(parseFloat(a)) && !isNaN(parseFloat(b))) {
                            return parseFloat(a) - parseFloat(b);
                        }
                        return a.localeCompare(b);
                    });
                }

                tempFilterDefinitionsMap.set(headerText, {
                    name: headerText,
                    originalKey: headerText, // For compare table, originalKey is the same as name
                    visibleIndex: visibleColIndex,
                    values: columnValues
                });
            });

            // Manually add "Body Tilt" filter definition for the combined table
            const bodyTiltKey = 'Body Tilt';
            tempFilterDefinitionsMap.set(bodyTiltKey, {
                name: bodyTiltKey,
                originalKey: bodyTiltKey,
                visibleIndex: -1, // Conceptual filter, no direct column
                values: ['Shoulder Tilt', 'Hip Tilt'] // Values for the dropdown
            });

            // Reconstruct the ordered keys for compare table
            const orderedKeys = [];
            ['Pose Class', 'Body Tilt', 'Improvement Status'].forEach(key => {
                if (tempFilterDefinitionsMap.has(key)) {
                    orderedKeys.push(key);
                }
            });

            // Add any other remaining filter definitions (shouldn't be any based on defined columns)
            tempFilterDefinitionsMap.forEach(def => {
                if (!orderedKeys.includes(def.originalKey)) {
                    orderedKeys.push(def.originalKey);
                }
            });

            tableColumnFilterDefinitions = orderedKeys.map(key => tempFilterDefinitionsMap.get(key));

            // Initialize tableFilterStates
            tableColumnFilterDefinitions.forEach(col => {
                if (!tableFilterStates[col.originalKey]) {
                    tableFilterStates[col.originalKey] = {};
                    col.values.forEach(val => {
                        tableFilterStates[col.originalKey][val.toLowerCase()] = true; // Default to selected
                    });
                }
            });
        }

        function renderCompareTableFilterOptions() {
            tableFilterOptionsContainer.innerHTML = '';
            tableFilterOptionsContainer.className = 'flex flex-wrap -mx-2';

            tableColumnFilterDefinitions.forEach(col => {
                const columnSection = document.createElement('div');
                columnSection.className = 'w-1/2 px-2 mb-4 border-b pb-2';

                const sectionHeader = document.createElement('h3');
                sectionHeader.className = 'font-semibold mb-2 text-gray-700 md:text-sm text-xs';
                sectionHeader.textContent = col.name;
                columnSection.appendChild(sectionHeader);

                const selectAllDiv = document.createElement('div');
                selectAllDiv.className = 'flex items-center mb-1';
                const selectAllInput = document.createElement('input');
                selectAllInput.type = 'checkbox';
                selectAllInput.id = `table-select-all-${col.originalKey.replace(/\s+/g, '-')}`;
                selectAllInput.className = 'form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 select-all-checkbox';
                selectAllInput.dataset.originalKey = col.originalKey;

                const selectAllLabel = document.createElement('label');
                selectAllLabel.htmlFor = `table-select-all-${col.originalKey.replace(/\s+/g, '-')}`;
                selectAllLabel.className = 'text-gray-700 md:text-sm text-xs cursor-pointer';
                selectAllLabel.textContent = 'Select All';
                selectAllDiv.appendChild(selectAllInput);
                selectAllDiv.appendChild(selectAllLabel);
                columnSection.appendChild(selectAllDiv);

                let allValuesChecked = true;

                col.values.forEach(value => {
                    const optionDiv = document.createElement('div');
                    optionDiv.className = 'flex items-center ml-4 mb-1';
                    const input = document.createElement('input');
                    input.type = 'checkbox';
                    input.id = `table-filter-${col.originalKey.replace(/\s+/g, '-')}-${value.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '')}`;
                    input.className = `form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 column-value-checkbox column-${col.originalKey.replace(/\s+/g, '-')}`;
                    input.value = value;
                    input.checked = tableFilterStates[col.originalKey] && tableFilterStates[col.originalKey][value.toLowerCase()] !== undefined
                                    ? tableFilterStates[col.originalKey][value.toLowerCase()]
                                    : true;

                    if (!input.checked) {
                        allValuesChecked = false;
                    }

                    input.dataset.originalKey = col.originalKey;

                    const label = document.createElement('label');
                    label.htmlFor = input.id;
                    label.className = 'text-gray-700 md:text-sm text-xs cursor-pointer';
                    label.textContent = value;
                    optionDiv.appendChild(input);
                    optionDiv.appendChild(label);
                    columnSection.appendChild(optionDiv);
                });

                selectAllInput.checked = allValuesChecked;
                tableFilterOptionsContainer.appendChild(columnSection);
            });

            document.querySelectorAll('#table-filter-panel .select-all-checkbox').forEach(checkbox => {
                checkbox.addEventListener('change', function() {
                    const originalKey = this.dataset.originalKey;
                    const isChecked = this.checked;
                    document.querySelectorAll(`#table-filter-panel .column-${originalKey.replace(/\s+/g, '-')}`).forEach(valCheckbox => {
                        valCheckbox.checked = isChecked;
                    });
                });
            });

            document.querySelectorAll('#table-filter-panel .column-value-checkbox').forEach(checkbox => {
                checkbox.addEventListener('change', function() {
                    const originalKey = this.dataset.originalKey;
                    const selectAllCheckbox = document.getElementById(`table-select-all-${originalKey.replace(/\s+/g, '-')}`);
                    const allColumnCheckboxes = document.querySelectorAll(`#table-filter-panel .column-${originalKey.replace(/\s+/g, '-')}`);
                    const allChecked = Array.from(allColumnCheckboxes).every(cb => cb.checked);

                    selectAllCheckbox.checked = allChecked;
                });
            });
        }

        function applyCompareTableFilters() {
            const activeFilters = new Map();

            tableColumnFilterDefinitions.forEach(colDef => {
                tableFilterStates[colDef.originalKey] = {};
                const currentColumnCheckboxes = document.querySelectorAll(`#table-filter-panel .column-${colDef.originalKey.replace(/\s+/g, '-')}`);
                let selectedValuesCount = 0;
                currentColumnCheckboxes.forEach(checkbox => {
                    const isChecked = checkbox.checked;
                    tableFilterStates[colDef.originalKey][checkbox.value.toLowerCase()] = isChecked;
                    if (isChecked) {
                        selectedValuesCount++;
                    }
                });

                if (selectedValuesCount > 0 && selectedValuesCount < colDef.values.length) {
                    const selectedValues = new Set();
                    currentColumnCheckboxes.forEach(checkbox => {
                        if (checkbox.checked) {
                            selectedValues.add(checkbox.value.toLowerCase());
                        }
                    });
                    activeFilters.set(colDef.originalKey, selectedValues);
                }
            });

            // On the Compare Page, all columns are always visible.
            // The 'Body Tilt' filter affects ROWS based on the content of the 'Body Tilt' column.
            tableHeaders.forEach((header, index) => {
                 header.style.display = ''; // Ensure all headers are visible
                 tableRows.forEach(row => {
                     const cell = row.querySelectorAll('td')[index];
                     if (cell) cell.style.display = ''; // Ensure all cells are visible by default
                 });
            });


            // --- Handle Row Visibility for Combined Table (for all other filters including Body Tilt) ---
            let bodyTiltActualColIndex = -1; // Index of the 'Body Tilt' column itself

            tableHeaders.forEach((header, index) => {
                if (header.textContent.trim() === 'Body Tilt') {
                    bodyTiltActualColIndex = index;
                }
            });

            tableRows.forEach(row => {
                let rowVisible = true;
                const cells = row.querySelectorAll('td');

                // Apply Body Tilt row filtering
                const bodyTiltFilterDef = tableColumnFilterDefinitions.find(col => col.originalKey === 'Body Tilt');
                if (bodyTiltFilterDef && tableFilterStates['Body Tilt'] && bodyTiltActualColIndex !== -1) {
                    const isShoulderSelected = tableFilterStates['Body Tilt']['shoulder tilt'];
                    const isHipSelected = tableFilterStates['Body Tilt']['hip tilt'];

                    // Get the text content of the 'Body Tilt' cell for the current row
                    const bodyTiltCellValue = cells[bodyTiltActualColIndex] ? cells[bodyTiltActualColIndex].textContent.trim().toLowerCase() : '';

                    // Determine if the row's 'Body Tilt' cell contains Shoulder Tilt or Hip Tilt data
                    // Based on the image, a cell might contain "Shoulder Tilt", "Hip Tilt", "Both", or be empty/null for "None".
                    const hasShoulderDataInCell = bodyTiltCellValue.includes('shoulder tilt') || bodyTiltCellValue.includes('both');
                    const hasHipDataInCell = bodyTiltCellValue.includes('hip tilt') || bodyTiltCellValue.includes('both');
                    // The 'hasNoTiltDataInCell' is implicitly handled by the conditions below.
                    // If neither 'shoulder tilt' nor 'hip tilt' nor 'both' is present,
                    // and neither filter is selected, then it's visible. If filters are selected, it's not.

                    // Logic for row visibility based on 'Body Tilt' filter selection
                    if (isShoulderSelected && !isHipSelected) {
                        // Only Shoulder Tilt selected: row is visible only if it has Shoulder Tilt data
                        rowVisible = hasShoulderDataInCell;
                    } else if (!isShoulderSelected && isHipSelected) {
                        // Only Hip Tilt selected: row is visible only if it has Hip Tilt data
                        rowVisible = hasHipDataInCell;
                    } else if (isShoulderSelected && isHipSelected) {
                        // Both selected: row is visible if it has either Shoulder Tilt or Hip Tilt data
                        rowVisible = hasShoulderDataInCell || hasHipDataInCell;
                    } else { // Neither 'Shoulder Tilt' nor 'Hip Tilt' is selected (i.e., 'Select All' unchecked for both)
                        // Hide rows that have ANY Body Tilt data (show only rows that explicitly have NO tilt data in the cell)
                        // This implies showing rows where the 'Body Tilt' cell is effectively empty or indicates no specific tilt.
                        rowVisible = !hasShoulderDataInCell && !hasHipDataInCell;
                    }
                }


                // Apply other filters only if the row is still visible after Body Tilt filter
                if (rowVisible) {
                    for (const colDef of tableColumnFilterDefinitions) {
                        // Skip 'Body Tilt' as its row filtering is already handled above.
                        // Other filters like 'Pose Class' and 'Improvement Status' should apply.
                        if (colDef.originalKey === 'Body Tilt') {
                            continue;
                        }

                        if (activeFilters.has(colDef.originalKey)) {
                            const selectedValues = activeFilters.get(colDef.originalKey);

                            if (colDef.visibleIndex !== -1 && cells[colDef.visibleIndex]) {
                                const cellValue = cells[colDef.visibleIndex].textContent.trim().toLowerCase();
                                if (!selectedValues.has(cellValue)) {
                                    rowVisible = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                row.style.display = rowVisible ? '' : 'none';
            });

            tableFilterPanel.classList.add('hidden');
        }

        function clearCompareTableFilters() {
            tableColumnFilterDefinitions.forEach(col => {
                tableFilterStates[col.originalKey] = {};
                col.values.forEach(val => {
                    tableFilterStates[col.originalKey][val.toLowerCase()] = true;
                });
            });

            document.querySelectorAll('#table-filter-panel .select-all-checkbox').forEach(checkbox => {
                checkbox.checked = true;
            });
            document.querySelectorAll('#table-filter-panel .column-value-checkbox').forEach(checkbox => {
                checkbox.checked = true;
            });

            // Ensure all columns are visible (revert from potential prior state if any bug caused them to hide)
            tableHeaders.forEach((header, index) => {
                header.style.display = '';
                tableRows.forEach(row => {
                    const cell = row.querySelectorAll('td')[index];
                    if (cell) cell.style.display = '';
                });
            });

            tableRows.forEach(row => {
                row.style.display = ''; // Make all rows visible
            });
            tableFilterPanel.classList.add('hidden');
        }


        // --- Helper Functions for Compare Page Chart ---

        // This function should be called by dashboard_compareSwings.js
        // when the chart is created/updated.
        window.setChartInstance = function(chartInstance) {
            currentChartInstance = chartInstance;
            createChartFilters();
        };

        function createChartFilters() {
            // Only proceed if a chart instance is available and the container exists
            if (!currentChartInstance || !chartFiltersContainer) {
                console.log("Chart instance or chart filters container not available. Cannot create chart filters.");
                return;
            }

            chartFiltersContainer.innerHTML = ''; // Clear existing filters
            chartFiltersContainer.className = 'mb-4 flex flex-wrap gap-4'; // Apply styling to the container

            currentChartInstance.data.datasets.forEach((dataset, index) => {
                const filterItem = document.createElement('div');
                filterItem.className = 'flex items-center space-x-2'; // Flex container for checkbox, colorbox, label

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `chart-filter-${index}`;
                checkbox.checked = chartFilterStates[dataset.label] !== undefined
                                    ? chartFilterStates[dataset.label]
                                    : !dataset.hidden; // Maintain state or default to visible
                checkbox.value = dataset.label;
                checkbox.dataset.datasetIndex = index;
                // Crucial Tailwind CSS classes for visual checkbox appearance
                checkbox.className = 'form-checkbox h-4 w-4 text-indigo-600 rounded';

                const colorBox = document.createElement('span');
                colorBox.className = 'w-4 h-4 rounded-full inline-block';
                colorBox.style.backgroundColor = dataset.borderColor || dataset.backgroundColor;

                const label = document.createElement('label');
                label.htmlFor = `chart-filter-${index}`;
                label.textContent = dataset.label;
                // Tailwind CSS classes for label styling and pointer cursor
                label.className = 'text-sm text-gray-700 cursor-pointer';

                checkbox.addEventListener('change', (event) => {
                    const targetIndex = parseInt(event.target.dataset.datasetIndex);
                    const isChecked = event.target.checked;
                    currentChartInstance.data.datasets[targetIndex].hidden = !isChecked;
                    chartFilterStates[dataset.label] = isChecked;
                    currentChartInstance.update();
                });

                filterItem.appendChild(checkbox);
                filterItem.appendChild(colorBox);
                filterItem.appendChild(label);
                chartFiltersContainer.appendChild(filterItem);

                if (chartFilterStates[dataset.label] === undefined) {
                    chartFilterStates[dataset.label] = !dataset.hidden;
                }
            });
        }


        // --- Event Listeners for Compare Page ---

        // Table Filter Button
        if (tableFilterDropdownBtn) {
            tableFilterDropdownBtn.addEventListener('click', function(event) {
                event.stopPropagation();
                const isHidden = tableFilterPanel.classList.contains('hidden');
                if (isHidden) {
                    prepareCompareTableFilterData(); // Re-prepare data as table content can change
                    renderCompareTableFilterOptions();
                    tableFilterPanel.classList.remove('hidden');
                } else {
                    tableFilterPanel.classList.add('hidden');
                }
            });

            document.addEventListener('click', function(event) {
                if (tableFilterPanel && !tableFilterPanel.contains(event.target) && !tableFilterDropdownBtn.contains(event.target)) {
                    tableFilterPanel.classList.add('hidden');
                }
            });

            if (tableFilterPanel) {
                tableFilterPanel.addEventListener('click', function(event) {
                    event.stopPropagation();
                });
            }

            if (tableApplyFiltersBtn) tableApplyFiltersBtn.addEventListener('click', applyCompareTableFilters);
            if (tableClearFiltersBtn) tableClearFiltersBtn.addEventListener('click', clearCompareTableFilters);
        }

        // Handle tab switching to refresh table data and filters
        const comparisonTabs = document.querySelectorAll('nav[aria-label="Comparison Tabs"] button');
        comparisonTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Deactivate all tab contents
                document.querySelectorAll('.tab-content').forEach(content => content.classList.add('hidden'));
                // Activate the selected tab content
                const targetTabId = `tab-${tab.dataset.tab}`;
                document.getElementById(targetTabId).classList.remove('hidden');

                // Update active tab styling
                comparisonTabs.forEach(t => {
                    t.classList.remove('text-blue-600', 'border-blue-600');
                    t.classList.add('text-gray-500', 'hover:text-blue-600', 'hover:border-blue-600', 'border-transparent');
                });
                tab.classList.remove('text-gray-500', 'hover:text-blue-600', 'hover:border-blue-600', 'border-transparent');
                tab.classList.add('text-blue-600', 'border-b-2', 'border-blue-600');

                // If switching to table view, re-prepare data and apply filters
                const isTableView = document.getElementById('table-view').classList.contains('hidden') === false;
                if (isTableView) {
                    // Ensure current table elements are picked up correctly
                    getCompareTableElements();
                    prepareCompareTableFilterData();
                    applyCompareTableFilters(); // Re-apply filters based on current state
                }

                // If switching to chart view, recreate chart filters
                const isChartView = document.getElementById('chart-view').classList.contains('hidden') === false;
                if (isChartView && currentChartInstance) { // Only attempt to create if chart instance exists
                    createChartFilters(); // Recreate chart filters based on current chart instance
                }
            });
        });

        // Initialize table filters on page load if table is visible
        if (document.getElementById('table-view') && !document.getElementById('table-view').classList.contains('hidden')) {
            getCompareTableElements();
            prepareCompareTableFilterData();
            applyCompareTableFilters();
        }


    } else {
        // --- Original Results Page Filtering Logic ---
        // (Your existing code for the dashboard results page - NO CHANGES HERE)

        const table = document.getElementById('data-table');
        if (!table) {
            console.warn("Table with ID 'data-table' not found. Table filtering will not work.");
            return;
        }

        const rows = Array.from(table.querySelectorAll('tbody tr'));
        const tableHeaders = Array.from(table.querySelectorAll('thead th'));

        const filterDropdownBtn = document.getElementById('filter-dropdown-btn');
        const filterPanel = document.getElementById('filter-panel');
        const filterOptionsContainer = document.getElementById('filter-options-container');
        const applyFiltersBtn = document.getElementById('apply-filters-btn');
        const clearFiltersBtn = document.getElementById('clear-filters-btn');

        let columnFilterDefinitions = [];
        let headerNameToVisibleIndexMap = new Map();
        let filterStates = {};

        // --- Helper Functions (Original) ---
        function prepareFilterData() {
            columnFilterDefinitions = [];
            headerNameToVisibleIndexMap = new Map();

            const visibleHeaderTexts = tableHeaders.map(th => th.textContent.trim());

            visibleHeaderTexts.forEach((headerText, visibleColIndex) => {
                headerNameToVisibleIndexMap.set(headerText, visibleColIndex);
            });

            const columnsToExcludeFromFilterUI = [
                'Time Frame', 'Shoulder Tilt', 'Hip Tilt',
            ];

            const tempFilterDefinitionsMap = new Map();

            visibleHeaderTexts.forEach((headerText, visibleColIndex) => {
                let originalColKey = headerText;
                if (headerText === 'Status') {
                    originalColKey = 'Overall Status';
                }

                if (columnsToExcludeFromFilterUI.includes(headerText)) {
                    return;
                }

                const uniqueValues = new Set();
                rows.forEach(row => {
                    const cell = row.querySelectorAll('td')[visibleColIndex];
                    if (cell) {
                        uniqueValues.add(cell.textContent.trim());
                    }
                });

                let columnValues = Array.from(uniqueValues).filter(val => val !== '');

                if (originalColKey.endsWith('Status') || originalColKey === 'Overall Status') {
                    columnValues = ['Good', 'Bad', 'Very Bad'];
                } else if (originalColKey === 'Pose Class') {
                    // CORRECT SORTING LOGIC FOR POSE CLASS (repeated for original page)
                    columnValues.sort((a, b) => {
                        const numA = parseInt(a.substring(1));
                        const numB = parseInt(b.substring(1));
                        return numA - numB;
                    });
                } else {
                    columnValues.sort((a, b) => {
                        if (!isNaN(parseFloat(a)) && !isNaN(parseFloat(b))) {
                            return parseFloat(a) - parseFloat(b);
                        }
                        return a.localeCompare(b);
                    });
                }

                tempFilterDefinitionsMap.set(originalColKey, {
                    name: headerText,
                    originalKey: originalColKey,
                    visibleIndex: visibleColIndex,
                    values: columnValues
                });
            });

            const bodyTiltKey = 'Body Tilt';
            tempFilterDefinitionsMap.set(bodyTiltKey, {
                name: bodyTiltKey,
                originalKey: bodyTiltKey,
                visibleIndex: -1,
                values: ['Shoulder', 'Hip']
            });

            const orderedKeys = [];
            tableHeaders.forEach(th => {
                let headerText = th.textContent.trim();
                if (headerText === 'Status') {
                    headerText = 'Overall Status';
                }

                if (!columnsToExcludeFromFilterUI.includes(headerText) && tempFilterDefinitionsMap.has(headerText)) {
                    orderedKeys.push(headerText);
                }
            });

            let insertAfterKey = 'Pose Class';
            let insertIndex = orderedKeys.indexOf(insertAfterKey);

            if (insertIndex !== -1) {
                orderedKeys.splice(insertIndex + 1, 0, bodyTiltKey);
            } else {
                orderedKeys.push(bodyTiltKey);
            }

            columnFilterDefinitions = orderedKeys.map(key => tempFilterDefinitionsMap.get(key));

            columnFilterDefinitions.forEach(col => {
                if (!filterStates[col.originalKey]) {
                    filterStates[col.originalKey] = {};
                    col.values.forEach(val => {
                        filterStates[col.originalKey][val.toLowerCase()] = true;
                    });
                }
            });
        }

        function renderFilterOptions() {
            filterOptionsContainer.innerHTML = '';
            filterOptionsContainer.className = 'flex flex-wrap -mx-2';

            columnFilterDefinitions.forEach(col => {
                const columnSection = document.createElement('div');
                columnSection.className = 'w-1/2 px-2 mb-4 border-b pb-2';

                const sectionHeader = document.createElement('h3');
                sectionHeader.className = 'font-semibold mb-2 text-gray-700 md:text-sm text-xs';
                sectionHeader.textContent = col.name;
                columnSection.appendChild(sectionHeader);

                const selectAllDiv = document.createElement('div');
                selectAllDiv.className = 'flex items-center mb-1';
                const selectAllInput = document.createElement('input');
                selectAllInput.type = 'checkbox';
                selectAllInput.id = `select-all-${col.originalKey.replace(/\s+/g, '-')}`;
                selectAllInput.className = 'form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 select-all-checkbox';
                selectAllInput.dataset.originalKey = col.originalKey;

                const selectAllLabel = document.createElement('label');
                selectAllLabel.htmlFor = `select-all-${col.originalKey.replace(/\s+/g, '-')}`;
                selectAllLabel.className = 'text-gray-700 md:text-sm text-xs cursor-pointer';
                selectAllLabel.textContent = 'Select All';
                selectAllDiv.appendChild(selectAllInput);
                selectAllDiv.appendChild(selectAllLabel);
                columnSection.appendChild(selectAllDiv);

                let allValuesChecked = true;

                col.values.forEach(value => {
                    const optionDiv = document.createElement('div');
                    optionDiv.className = 'flex items-center ml-4 mb-1';
                    const input = document.createElement('input');
                    input.type = 'checkbox';
                    input.id = `filter-${col.originalKey.replace(/\s+/g, '-')}-${value.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '')}`;
                    input.className = `form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 column-value-checkbox column-${col.originalKey.replace(/\s+/g, '-')}`;
                    input.value = value;
                    input.checked = filterStates[col.originalKey] && filterStates[col.originalKey][value.toLowerCase()] !== undefined
                                    ? filterStates[col.originalKey][value.toLowerCase()]
                                    : true;

                    if (!input.checked) {
                        allValuesChecked = false;
                    }

                    input.dataset.originalKey = col.originalKey;

                    const label = document.createElement('label');
                    label.htmlFor = input.id;
                    label.className = 'text-gray-700 md:text-sm text-xs cursor-pointer';
                    label.textContent = value;
                    optionDiv.appendChild(input);
                    optionDiv.appendChild(label);
                    columnSection.appendChild(optionDiv);
                });

                selectAllInput.checked = allValuesChecked;

                filterOptionsContainer.appendChild(columnSection);
            });

            document.querySelectorAll('.select-all-checkbox').forEach(checkbox => {
                checkbox.addEventListener('change', function() {
                    const originalKey = this.dataset.originalKey;
                    const isChecked = this.checked;
                    document.querySelectorAll(`.column-${originalKey.replace(/\s+/g, '-')}`).forEach(valCheckbox => {
                        valCheckbox.checked = isChecked;
                    });
                });
            });

            document.querySelectorAll('.column-value-checkbox').forEach(checkbox => {
                checkbox.addEventListener('change', function() {
                    const originalKey = this.dataset.originalKey;
                    const selectAllCheckbox = document.getElementById(`select-all-${originalKey.replace(/\s+/g, '-')}`);
                    const allColumnCheckboxes = document.querySelectorAll(`.column-${originalKey.replace(/\s+/g, '-')}`);
                    const allChecked = Array.from(allColumnCheckboxes).every(cb => cb.checked);

                    selectAllCheckbox.checked = allChecked;
                });
            });
        }

        function applyFilters() {
            const activeFilters = new Map();

            columnFilterDefinitions.forEach(colDef => {
                filterStates[colDef.originalKey] = {};
                const currentColumnCheckboxes = document.querySelectorAll(`.column-${colDef.originalKey.replace(/\s+/g, '-')}`);
                let selectedValuesCount = 0;
                currentColumnCheckboxes.forEach(checkbox => {
                    const isChecked = checkbox.checked;
                    filterStates[colDef.originalKey][checkbox.value.toLowerCase()] = isChecked;
                    if (isChecked) {
                        selectedValuesCount++;
                    }
                });

                if (selectedValuesCount > 0 && selectedValuesCount < colDef.values.length) {
                    const selectedValues = new Set();
                    currentColumnCheckboxes.forEach(checkbox => {
                        if (checkbox.checked) {
                            selectedValues.add(checkbox.value.toLowerCase());
                        }
                    });
                    activeFilters.set(colDef.originalKey, selectedValues);
                }
            });

            // This is the COLUMN visibility logic specifically for the ORIGINAL RESULTS PAGE
            const bodyTiltFilterDef = columnFilterDefinitions.find(col => col.originalKey === 'Body Tilt');
            if (bodyTiltFilterDef && filterStates['Body Tilt']) {
                const isShoulderSelected = filterStates['Body Tilt']['shoulder'];
                const isHipSelected = filterStates['Body Tilt']['hip'];

                const shoulderRelatedCols = ['Shoulder Tilt', 'Shoulder Tilt Status'];
                const hipRelatedCols = ['Hip Tilt', 'Hip Tilt Status'];

                tableHeaders.forEach((header, index) => {
                    const headerText = header.textContent.trim();
                    let showColumn = true;

                    if (shoulderRelatedCols.includes(headerText)) {
                        showColumn = isShoulderSelected;
                    } else if (hipRelatedCols.includes(headerText)) {
                        showColumn = isHipSelected;
                    }

                    header.style.display = showColumn ? '' : 'none';
                    rows.forEach(row => {
                        const cell = row.querySelectorAll('td')[index];
                        if (cell) {
                            cell.style.display = showColumn ? '' : 'none';
                        }
                    });
                });
            } else {
                // If Body Tilt filter is not active, ensure all related columns are visible by default
                const shoulderRelatedCols = ['Shoulder Tilt', 'Shoulder Tilt Status'];
                const hipRelatedCols = ['Hip Tilt', 'Hip Tilt Status'];
                tableHeaders.forEach((header, index) => {
                     const headerText = header.textContent.trim();
                     if (shoulderRelatedCols.includes(headerText) || hipRelatedCols.includes(headerText)) {
                         header.style.display = '';
                         rows.forEach(row => {
                             const cell = row.querySelectorAll('td')[index];
                             if (cell) cell.style.display = '';
                         });
                     }
                });
            }

            rows.forEach(row => {
                let rowVisible = true;
                const cells = row.querySelectorAll('td');

                for (const colDef of columnFilterDefinitions) {
                    // Skip 'Body Tilt' as its primary logic for Results Page is column visibility.
                    // Row filtering for Shoulder/Hip Tilt values isn't directly controlled by this filter for the results table.
                    if (colDef.originalKey === 'Body Tilt') {
                        continue;
                    }

                    if (activeFilters.has(colDef.originalKey)) {
                        const selectedValues = activeFilters.get(colDef.originalKey);

                        if (colDef.visibleIndex !== -1 && cells[colDef.visibleIndex]) {
                            const cellValue = cells[colDef.visibleIndex].textContent.trim().toLowerCase();
                            if (!selectedValues.has(cellValue)) {
                                rowVisible = false;
                                break;
                            }
                        }
                    }
                }
                row.style.display = rowVisible ? '' : 'none';
            });

            filterPanel.classList.add('hidden');
        }

        function clearFilters() {
            columnFilterDefinitions.forEach(col => {
                filterStates[col.originalKey] = {};
                col.values.forEach(val => {
                    filterStates[col.originalKey][val.toLowerCase()] = true;
                });
            });

            document.querySelectorAll('.select-all-checkbox').forEach(checkbox => {
                checkbox.checked = true;
            });
            document.querySelectorAll('.column-value-checkbox').forEach(checkbox => {
                checkbox.checked = true;
            });

            // Ensure all columns are visible on clear for the Results Page
            tableHeaders.forEach((header, index) => {
                header.style.display = '';
                rows.forEach(row => {
                    const cell = row.querySelectorAll('td')[index];
                    if (cell) cell.style.display = '';
                });
            });

            rows.forEach(row => {
                row.style.display = '';
            });
            filterPanel.classList.add('hidden');
        }

        // --- Event Listeners (Original) ---
        filterDropdownBtn.addEventListener('click', function(event) {
            event.stopPropagation();
            const isHidden = filterPanel.classList.contains('hidden');
            if (isHidden) {
                prepareFilterData();
                renderFilterOptions();
                filterPanel.classList.remove('hidden');
            } else {
                filterPanel.classList.add('hidden');
            }
        });

        document.addEventListener('click', function(event) {
            if (!filterPanel.contains(event.target) && !filterDropdownBtn.contains(event.target)) {
                filterPanel.classList.add('hidden');
            }
        });

        filterPanel.addEventListener('click', function(event) {
            event.stopPropagation();
        });

        applyFiltersBtn.addEventListener('click', applyFilters);
        clearFiltersBtn.addEventListener('click', clearFilters);

        // Initial setup for original page
        prepareFilterData();
    }
});

// IMPORTANT: You need to ensure your dashboard_compareSwings.js
// exposes the Chart.js instance using this function:
//
// In dashboard_compareSwings.js, after you create or update your chart:
//
// const tiltChartCtx = document.getElementById('tiltChart').getContext('2d');
// if (window.currentChart) { // Destroy existing chart if it exists
//     window.currentChart.destroy();
// }
// window.currentChart = new Chart(tiltChartCtx, {
//    // ... your chart configuration ...
// });
// if (window.setChartInstance) { // Check if the function exists
//    window.setChartInstance(window.currentChart);
// }