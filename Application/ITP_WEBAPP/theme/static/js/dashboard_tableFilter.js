document.addEventListener('DOMContentLoaded', function () {
    const table = document.getElementById('data-table');
    if (!table) {
        console.warn("Table with ID 'data-table' not found. Table filtering will not work.");
        return;
    }

    const rows = Array.from(table.querySelectorAll('tbody tr'));
    const tableHeaders = Array.from(table.querySelectorAll('thead th')); // Convert to array for easier indexing

    const filterDropdownBtn = document.getElementById('filter-dropdown-btn');
    const filterPanel = document.getElementById('filter-panel');
    const filterOptionsContainer = document.getElementById('filter-options-container');
    const applyFiltersBtn = document.getElementById('apply-filters-btn');
    const clearFiltersBtn = document.getElementById('clear-filters-btn');

    let columnFilterDefinitions = [];
    let headerNameToVisibleIndexMap = new Map();
    let filterStates = {};

    // --- Helper Functions ---

    function prepareFilterData() {
        columnFilterDefinitions = [];
        headerNameToVisibleIndexMap = new Map();

        const visibleHeaderTexts = tableHeaders.map(th => th.textContent.trim());

        // Step 1: Populate headerNameToVisibleIndexMap for quick lookups
        visibleHeaderTexts.forEach((headerText, visibleColIndex) => {
            headerNameToVisibleIndexMap.set(headerText, visibleColIndex);
        });

        // Columns that should NOT have their own filter UI (e.g., numerical values)
        const columnsToExcludeFromFilterUI = [
            'Time Frame',
            'Shoulder Tilt',
            'Hip Tilt',
        ];

        // Create a temporary map to hold filter definitions by originalKey for easy access
        const tempFilterDefinitionsMap = new Map();

        // Populate the map with standard filter definitions
        visibleHeaderTexts.forEach((headerText, visibleColIndex) => {
            let originalColKey = headerText;
            if (headerText === 'Status') {
                originalColKey = 'Overall Status'; // Handle rename for internal consistency
            }

            if (columnsToExcludeFromFilterUI.includes(headerText)) {
                return; // Skip creating a filter definition for these specific columns
            }

            const uniqueValues = new Set();
            rows.forEach(row => {
                const cell = row.querySelectorAll('td')[visibleColIndex];
                if (cell) {
                    uniqueValues.add(cell.textContent.trim());
                }
            });

            let columnValues = Array.from(uniqueValues).filter(val => val !== '');

            // Special handling for status columns to ensure consistent values
            if (originalColKey.endsWith('Status') || originalColKey === 'Overall Status') {
                columnValues = ['Good', 'Bad', 'Very Bad']; // Ensure full set of status options
            } else if (originalColKey === 'Pose Class') { // <-- NEW: Custom sorting for Pose Class
                columnValues.sort((a, b) => {
                    const numA = parseInt(a.substring(1)); // Extract number from "P1", "P10"
                    const numB = parseInt(b.substring(1)); // Extract number from "P1", "P10"
                    return numA - numB;
                });
            } else { // Existing generic sorting for other columns
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

        // Manually add "Body Tilt" filter definition if it doesn't exist, or update it
        const bodyTiltKey = 'Body Tilt';
        tempFilterDefinitionsMap.set(bodyTiltKey, {
            name: bodyTiltKey,
            originalKey: bodyTiltKey,
            visibleIndex: -1, // Conceptual filter, no direct column
            values: ['Shoulder', 'Hip'] // Values for the dropdown
        });

        // Step 2: Build columnFilterDefinitions, inserting Body Tilt at the desired position
        // This approach maintains the original order of other filters and injects Body Tilt
        const orderedKeys = [];

        // First, build a list of keys in their original table order, excluding 'Body Tilt'
        tableHeaders.forEach(th => {
            let headerText = th.textContent.trim();
            if (headerText === 'Status') { // Handle "Overall Status" rename
                headerText = 'Overall Status';
            }

            // Exclude numerical columns and ensure only filterable items are considered for order
            if (!columnsToExcludeFromFilterUI.includes(headerText) && tempFilterDefinitionsMap.has(headerText)) {
                orderedKeys.push(headerText);
            }
        });

        // Find the index to insert 'Body Tilt'
        let insertAfterKey = 'Pose Class';
        let insertIndex = orderedKeys.indexOf(insertAfterKey);

        if (insertIndex !== -1) {
            // Insert Body Tilt after 'Pose Class'
            orderedKeys.splice(insertIndex + 1, 0, bodyTiltKey);
        } else {
            // If 'Pose Class' isn't found, just add Body Tilt at the end (fallback)
            orderedKeys.push(bodyTiltKey);
        }

        // Now, populate columnFilterDefinitions based on this custom order
        orderedKeys.forEach(key => {
            if (tempFilterDefinitionsMap.has(key)) {
                columnFilterDefinitions.push(tempFilterDefinitionsMap.get(key));
                tempFilterDefinitionsMap.delete(key); // Remove from map after adding
            }
        });

        // Add any remaining filters that weren't in the orderedKeys (shouldn't be many if any, now)
        tempFilterDefinitionsMap.forEach(def => {
            columnFilterDefinitions.push(def);
        });


        // Initialize filterStates for newly found/defined filters if not already present
        columnFilterDefinitions.forEach(col => {
            if (!filterStates[col.originalKey]) {
                filterStates[col.originalKey] = {};
                col.values.forEach(val => {
                    filterStates[col.originalKey][val.toLowerCase()] = true; // Default to selected
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
        const activeFilters = new Map(); // For row filtering (e.g., Overall Status, Shoulder Tilt Status)

        // Store the current state of checkboxes for persistence
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

            // If selected values are less than total values, add to active filters for actual filtering
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

        // --- Handle Column Visibility based on "Body Tilt" filter ---
        // This filter will control visibility of Shoulder Tilt, Shoulder Tilt Status, Hip Tilt, Hip Tilt Status
        const bodyTiltFilterDef = columnFilterDefinitions.find(col => col.originalKey === 'Body Tilt');
        if (bodyTiltFilterDef && filterStates['Body Tilt']) {
            const isShoulderSelected = filterStates['Body Tilt']['shoulder'];
            const isHipSelected = filterStates['Body Tilt']['hip'];

            const shoulderRelatedCols = ['Shoulder Tilt', 'Shoulder Tilt Status'];
            const hipRelatedCols = ['Hip Tilt', 'Hip Tilt Status'];

            tableHeaders.forEach((header, index) => {
                const headerText = header.textContent.trim();
                let showColumn = true; // Default to showing the column

                if (shoulderRelatedCols.includes(headerText)) {
                    showColumn = isShoulderSelected; // Show if 'Shoulder' is selected in Body Tilt filter
                } else if (hipRelatedCols.includes(headerText)) {
                    showColumn = isHipSelected; // Show if 'Hip' is selected in Body Tilt filter
                }
                // Other columns (e.g., Overall Status, ID) remain untouched by Body Tilt filter

                header.style.display = showColumn ? '' : 'none'; // Apply to header
                rows.forEach(row => {
                    const cell = row.querySelectorAll('td')[index];
                    if (cell) {
                        cell.style.display = showColumn ? '' : 'none'; // Apply to cells in that column
                    }
                });
            });
        } else {
            // If Body Tilt filter is not active, ensure all related columns are visible
            const shoulderRelatedCols = ['Shoulder Tilt', 'Shoulder Tilt Status'];
            const hipRelatedCols = ['Hip Tilt', 'Hip Tilt Status'];
            tableHeaders.forEach((header, index) => {
                 const headerText = header.textContent.trim();
                 if (shoulderRelatedCols.includes(headerText) || hipRelatedCols.includes(headerText)) {
                     header.style.display = ''; // Ensure header is visible
                     rows.forEach(row => {
                         const cell = row.querySelectorAll('td')[index];
                         if (cell) cell.style.display = ''; // Ensure cells are visible
                     });
                 }
            });
        }


        // --- Handle Row Visibility (for all other filters including individual status filters) ---
        rows.forEach(row => {
            let rowVisible = true;
            const cells = row.querySelectorAll('td');

            for (const colDef of columnFilterDefinitions) {
                // Skip Body Tilt, as its logic is solely for column visibility, not row filtering.
                if (colDef.originalKey === 'Body Tilt') {
                    continue;
                }

                if (activeFilters.has(colDef.originalKey)) {
                    const selectedValues = activeFilters.get(colDef.originalKey);

                    // Standard row filtering for all other columns (e.g., Overall Status, Shoulder Tilt Status, Hip Tilt Status)
                    if (colDef.visibleIndex !== -1 && cells[colDef.visibleIndex]) {
                        const cellValue = cells[colDef.visibleIndex].textContent.trim().toLowerCase();
                        if (!selectedValues.has(cellValue)) {
                            rowVisible = false;
                            break;
                        }
                    } else {
                        // This case can happen if a column is technically in columnFilterDefinitions
                        // but is not present in the visible table (e.g., if a header was missing).
                        // It's a fallback, but ideally all columns in colDef should have a visibleIndex.
                        rowVisible = false; // Hide row if its filter column is not found
                        break;
                    }
                }
            }
            row.style.display = rowVisible ? '' : 'none';
        });

        filterPanel.classList.add('hidden');
    }

    function clearFilters() {
        // Reset filterStates to all true (selected) for all columns
        columnFilterDefinitions.forEach(col => {
            filterStates[col.originalKey] = {};
            col.values.forEach(val => {
                filterStates[col.originalKey][val.toLowerCase()] = true;
            });
        });

        // Visually update the checkboxes to reflect the cleared state
        document.querySelectorAll('.select-all-checkbox').forEach(checkbox => {
            checkbox.checked = true;
        });
        document.querySelectorAll('.column-value-checkbox').forEach(checkbox => {
            checkbox.checked = true;
        });

        // Ensure all columns are visible when filters are cleared (including those controlled by Body Tilt)
        tableHeaders.forEach((header, index) => {
            header.style.display = '';
            rows.forEach(row => {
                const cell = row.querySelectorAll('td')[index];
                if (cell) cell.style.display = '';
            });
        });

        // Show all rows
        rows.forEach(row => {
            row.style.display = '';
        });
        filterPanel.classList.add('hidden');
    }

    // --- Event Listeners ---

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

    // Initial setup
    prepareFilterData();
});