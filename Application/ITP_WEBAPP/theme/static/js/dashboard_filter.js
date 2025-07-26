// A global container for our named event handlers to allow for easy removal.
const comparePageEventHandlers = {};

// Hoist these variables to a higher scope so their state persists across function calls
let tableColumnFilterDefinitions = [];
let tableFilterStates = {}; // This will persist across tab switches

function setupComparePageEventListeners() {
    // --- Helper function to get the currently active tab panel ---
    const getActivePanel = () => document.querySelector('.tab-content:not(.hidden)');

    // --- Helper function to get elements from the active tab ---
    const getCompareTableElements = () => {
        const activePanel = getActivePanel();
        if (!activePanel) return { tableRows: [], tableHeaders: [] };

        const table = activePanel.querySelector('table');
        if (!table) return { tableRows: [], tableHeaders: [] };

        return {
            tableRows: Array.from(table.querySelectorAll('tbody tr')),
            tableHeaders: Array.from(table.querySelectorAll('thead th'))
        };
    };

    function prepareCompareTableFilterData() {
        const { tableRows, tableHeaders } = getCompareTableElements();
        if (tableRows.length === 0) return;

        // Reset definitions, but preserve filter states
        tableColumnFilterDefinitions = [];
        const tempFilterDefinitionsMap = new Map();
        
        const visibleHeaderTexts = tableHeaders.map(th => th.textContent.trim());

        const columnsToExcludeFromFilterUI = ['Video 1', 'Video 2', 'Difference'];

        visibleHeaderTexts.forEach((headerText, visibleColIndex) => {
            if (columnsToExcludeFromFilterUI.includes(headerText)) return;

            const uniqueValues = new Set();
            tableRows.forEach(row => {
                const cell = row.querySelectorAll('td')[visibleColIndex];
                if (cell) uniqueValues.add(cell.textContent.trim());
            });

            let columnValues = Array.from(uniqueValues).filter(val => val !== '');
            if (headerText === 'Improvement Status') {
                // Define a specific order for statuses
                const statusOrder = ['Good', 'Bad', 'Very Bad', 'Neutral', '-'];
                columnValues.sort((a,b) => statusOrder.indexOf(a) - statusOrder.indexOf(b));
            } else if (headerText.trim() === 'Pose Class') {
                columnValues.sort((a, b) => parseInt(a.substring(1)) - parseInt(b.substring(1)));
            } else {
                columnValues.sort((a, b) => a.localeCompare(b));
            }
            
            tempFilterDefinitionsMap.set(headerText, {
                name: headerText, originalKey: headerText, visibleIndex: visibleColIndex, values: columnValues
            });
        });
        
        // Add "Body Tilt" definition if it doesn't exist from headers
        if (!tempFilterDefinitionsMap.has('Body Tilt')) {
             tempFilterDefinitionsMap.set('Body Tilt', {
                name: 'Body Tilt',
                originalKey: 'Body Tilt',
                visibleIndex: 1, // The typical index for this column
                values: ['Shoulder Tilt', 'Hip Tilt']
            });
        }
        
        const orderedKeys = ['Pose Class', 'Body Tilt', 'Improvement Status'];
        tableColumnFilterDefinitions = orderedKeys
            .map(key => tempFilterDefinitionsMap.get(key))
            .filter(Boolean); // Filter out any undefined entries

        // Initialize or preserve filter states
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
        const activePanel = getActivePanel();
        if (!activePanel) return;
        const container = activePanel.querySelector('#table-filter-options-container');
        if (!container) return;
        
        container.innerHTML = '';
        container.className = 'flex flex-wrap -mx-2';

        tableColumnFilterDefinitions.forEach(col => {
            const columnSection = document.createElement('div');
            columnSection.className = 'w-1/2 px-2 mb-4 border-b pb-2';
            
            // Sanitize key for use in IDs
            const safeKey = col.originalKey.replace(/\s+/g, '-');
            
            columnSection.innerHTML = `
                <h3 class="font-semibold mb-2 text-gray-700 md:text-sm text-xs">${col.name}</h3>
                <div class="flex items-center mb-1">
                    <input type="checkbox" id="table-select-all-${safeKey}" class="form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 select-all-checkbox" data-original-key="${col.originalKey}">
                    <label for="table-select-all-${safeKey}" class="text-gray-700 md:text-sm text-xs cursor-pointer">Select All</label>
                </div>
            `;
            
            let allValuesChecked = true;
            col.values.forEach(value => {
                const isChecked = tableFilterStates[col.originalKey]?.[value.toLowerCase()] ?? true;
                if (!isChecked) allValuesChecked = false;
                
                const optionDiv = document.createElement('div');
                optionDiv.className = 'flex items-center ml-4 mb-1';
                // Sanitize value for use in IDs
                const safeValue = value.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '');
                
                optionDiv.innerHTML = `
                    <input type="checkbox" id="table-filter-${safeKey}-${safeValue}" class="form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 column-value-checkbox column-${safeKey}" value="${value}" data-original-key="${col.originalKey}" ${isChecked ? 'checked' : ''}>
                    <label for="table-filter-${safeKey}-${safeValue}" class="text-gray-700 md:text-sm text-xs cursor-pointer">${value}</label>
                `;
                columnSection.appendChild(optionDiv);
            });

            const selectAllCheckbox = columnSection.querySelector('.select-all-checkbox');
            if(selectAllCheckbox) selectAllCheckbox.checked = allValuesChecked;
            
            container.appendChild(columnSection);
        });
    }

    function applyCompareTableFilters() {
        const { tableRows } = getCompareTableElements();
        if (tableRows.length === 0) return;

        const activePanel = getActivePanel();
        const activeFilters = new Map();
        
        tableColumnFilterDefinitions.forEach(colDef => {
            const checkboxes = activePanel.querySelectorAll(`#table-filter-panel .column-${colDef.originalKey.replace(/\s+/g, '-')}`);
            tableFilterStates[colDef.originalKey] = {};
            const selectedValues = new Set();
            
            checkboxes.forEach(checkbox => {
                const isChecked = checkbox.checked;
                tableFilterStates[colDef.originalKey][checkbox.value.toLowerCase()] = isChecked;
                if (isChecked) selectedValues.add(checkbox.value.toLowerCase());
            });

            // Only add to activeFilters if it's a partial selection
            if (selectedValues.size > 0 && selectedValues.size < colDef.values.length) {
                activeFilters.set(colDef.originalKey, selectedValues);
            }
        });

        tableRows.forEach(row => {
            let rowVisible = true;
            const cells = row.querySelectorAll('td');

            for (const colDef of tableColumnFilterDefinitions) {
                if (activeFilters.has(colDef.originalKey)) {
                    const cellValue = cells[colDef.visibleIndex]?.textContent.trim().toLowerCase();
                    if (!activeFilters.get(colDef.originalKey).has(cellValue)) {
                        rowVisible = false;
                        break;
                    }
                }
            }
            row.style.display = rowVisible ? '' : 'none';
        });

        const filterPanel = activePanel.querySelector('#table-filter-panel');
        if (filterPanel) filterPanel.classList.add('hidden');
    }

    function clearCompareTableFilters() {
        const { tableRows } = getCompareTableElements();
        tableColumnFilterDefinitions.forEach(col => {
            if (tableFilterStates[col.originalKey]) {
                col.values.forEach(val => {
                    tableFilterStates[col.originalKey][val.toLowerCase()] = true;
                });
            }
        });
        
        tableRows.forEach(row => row.style.display = '');
        
        const activePanel = getActivePanel();
        const filterPanel = activePanel.querySelector('#table-filter-panel');
        if (filterPanel) {
            renderCompareTableFilterOptions(); // Re-render to check all boxes
            filterPanel.classList.add('hidden');
        }
    }

    const activePanel = getActivePanel();
    if (!activePanel) return;

    // --- Define Event Handlers ---
    comparePageEventHandlers.handleFilterDropdownClick = (event) => {
        event.stopPropagation();
        const panel = getActivePanel();
        const filterPanel = panel.querySelector('#table-filter-panel');
        if (!filterPanel) return;
        
        if (filterPanel.classList.toggle('hidden')) return;

        prepareCompareTableFilterData();
        renderCompareTableFilterOptions();
    };

    comparePageEventHandlers.handleApplyFilters = () => applyCompareTableFilters();
    comparePageEventHandlers.handleClearFilters = () => clearCompareTableFilters();
    
    comparePageEventHandlers.handleDocClick = (event) => {
        const panel = getActivePanel();
        if(!panel) return;
        const filterPanel = panel.querySelector('#table-filter-panel');
        const dropdownBtn = panel.querySelector('#table-filter-dropdown-btn');
        if (filterPanel && !filterPanel.classList.contains('hidden') && !filterPanel.contains(event.target) && !dropdownBtn.contains(event.target)) {
            filterPanel.classList.add('hidden');
        }
    };

    // --- Remove Old Listeners and Attach New Ones to Active Elements ---
    // Remove from all to be safe
    document.querySelectorAll('.table-filter-dropdown-btn').forEach(btn => btn.removeEventListener('click', comparePageEventHandlers.handleFilterDropdownClick));
    document.querySelectorAll('.table-apply-filters-btn').forEach(btn => btn.removeEventListener('click', comparePageEventHandlers.handleApplyFilters));
    document.querySelectorAll('.table-clear-filters-btn').forEach(btn => btn.removeEventListener('click', comparePageEventHandlers.handleClearFilters));
    // Remove document-level listener before re-adding
    document.removeEventListener('click', comparePageEventHandlers.handleDocClick);

    const dropdownBtn = activePanel.querySelector('#table-filter-dropdown-btn');
    const applyBtn = activePanel.querySelector('#table-apply-filters-btn');
    const clearBtn = activePanel.querySelector('#table-clear-filters-btn');
    const filterPanel = activePanel.querySelector('#table-filter-panel');

    if (dropdownBtn) dropdownBtn.addEventListener('click', comparePageEventHandlers.handleFilterDropdownClick);
    if (applyBtn) applyBtn.addEventListener('click', comparePageEventHandlers.handleApplyFilters);
    if (clearBtn) clearBtn.addEventListener('click', comparePageEventHandlers.handleClearFilters);
    document.addEventListener('click', comparePageEventHandlers.handleDocClick);

    // Use event delegation for checkboxes inside the filter panel
    if (filterPanel) {
        // Remove old listener if it exists to prevent duplicates
        if(comparePageEventHandlers.filterPanelChangeListener) {
            filterPanel.removeEventListener('change', comparePageEventHandlers.filterPanelChangeListener);
        }
        
        comparePageEventHandlers.filterPanelChangeListener = (event) => {
            const key = event.target.dataset.originalKey;
            if (event.target.classList.contains('select-all-checkbox')) {
                const isChecked = event.target.checked;
                getActivePanel().querySelectorAll(`.column-${key.replace(/\s+/g, '-')}`).forEach(cb => cb.checked = isChecked);
            } else if (event.target.classList.contains('column-value-checkbox')) {
                const allCheckboxes = getActivePanel().querySelectorAll(`.column-${key.replace(/\s+/g, '-')}`);
                const allChecked = Array.from(allCheckboxes).every(cb => cb.checked);
                const selectAll = getActivePanel().querySelector(`#table-select-all-${key.replace(/\s+/g, '-')}`);
                if(selectAll) selectAll.checked = allChecked;
            }
        };
        filterPanel.addEventListener('change', comparePageEventHandlers.filterPanelChangeListener);
    }

    // Apply any existing filters to the newly visible table
    prepareCompareTableFilterData();
    applyCompareTableFilters();
}

document.addEventListener('DOMContentLoaded', function () {
    const isComparePage = document.getElementById('compare-page') !== null;

    if (isComparePage) {
        // Expose the setup function to the global scope so dashboard_compareSwings.js can call it.
        window.setupComparePageEventListeners = setupComparePageEventListeners;
    } else {
        // --- Original Results Page Filtering Logic ---
        // This is the original, unmodified code for the other dashboard page.
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
            if (filterPanel && !filterPanel.contains(event.target) && !filterDropdownBtn.contains(event.target)) {
                filterPanel.classList.add('hidden');
            }
        });

        filterPanel.addEventListener('click', function(event) {
            event.stopPropagation();
        });

        applyFiltersBtn.addEventListener('click', applyFilters);
        clearFiltersBtn.addEventListener('click', clearFilters);

        prepareFilterData();
    }
});