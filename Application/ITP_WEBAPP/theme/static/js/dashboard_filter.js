// A global container for our named event handlers to allow for easy removal.
const comparePageEventHandlers = {};
const dataSpaceEventHandlers = {}; // New container for Data Space handlers

// Hoist these variables to a higher scope so their state persists across function calls
// These are primarily used by the Compare Page logic.
let tableColumnFilterDefinitions = [];
let tableFilterStates = {}; // This will persist across tab switches for Compare Page

// --- COMPARE PAGE FILTERING LOGIC (Used for data-table inside tab-content) ---

// Helper function to get the currently active tab panel
const getActivePanel = () => document.querySelector('.tab-content:not(.hidden)');

// Helper function to get elements from the active tab
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

    const columnsToExcludeFromFilterUI = ['Video 1', 'Video 2', 'Difference'];
    const visibleHeaderTexts = tableHeaders.map(th => th.textContent.trim());

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
        // We must re-read states from the DOM if the panel is open, otherwise use existing state.
        const filterPanel = activePanel.querySelector('#table-filter-panel');
        if (filterPanel && !filterPanel.classList.contains('hidden')) {
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
        } else {
            // Use saved state if the panel is closed/absent
            const savedState = tableFilterStates[colDef.originalKey];
            const allValues = colDef.values.map(v => v.toLowerCase());
            const selectedValues = new Set(allValues.filter(val => savedState[val]));

            if (selectedValues.size > 0 && selectedValues.size < colDef.values.length) {
                 activeFilters.set(colDef.originalKey, selectedValues);
            }
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

function setupComparePageEventListeners() {
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
        if (filterPanel && !filterPanel.classList.contains('hidden') && !filterPanel.contains(event.target) && dropdownBtn && !dropdownBtn.contains(event.target)) {
            filterPanel.classList.add('hidden');
        }
    };

    // --- Remove Old Listeners and Attach New Ones to Active Elements ---
    // Remove from all to be safe (needed because tabs change the active panel)
    document.querySelectorAll('.table-filter-dropdown-btn').forEach(btn => btn.removeEventListener('click', comparePageEventHandlers.handleFilterDropdownClick));
    document.querySelectorAll('.table-apply-filters-btn').forEach(btn => btn.removeEventListener('click', comparePageEventHandlers.handleApplyFilters));
    document.querySelectorAll('.table-clear-filters-btn').forEach(btn => btn.removeEventListener('click', comparePageEventHandlers.handleClearFilters));
    // Remove document-level listener before re-adding, only if it exists
    if(comparePageEventHandlers.handleDocClick) document.removeEventListener('click', comparePageEventHandlers.handleDocClick);

    const dropdownBtn = activePanel.querySelector('#table-filter-dropdown-btn');
    const applyBtn = activePanel.querySelector('#table-apply-filters-btn');
    const clearBtn = activePanel.querySelector('#table-clear-filters-btn');
    const filterPanel = activePanel.querySelector('#table-filter-panel');

    if (dropdownBtn) dropdownBtn.addEventListener('click', comparePageEventHandlers.handleFilterDropdownClick);
    if (applyBtn) applyBtn.addEventListener('click', comparePageEventHandlers.handleApplyFilters);
    if (clearBtn) clearBtn.addEventListener('click', comparePageEventHandlers.handleClearFilters);
    document.addEventListener('click', comparePageEventHandlers.handleDocClick); // Re-add document click listener

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


// --- DATA SPACE FILTERING LOGIC (Used for video cards/table on data space page) ---

// Helper function to get relevant video elements and their metadata in Data Space
function getDataSpaceVideoElements() {
    // Get the currently active tab content (e.g., #tab1, #tab2)
    const activeTab = document.querySelector('.tab-content:not(.hidden)');
    if (!activeTab) return { videos: [] };

    const isListView = !activeTab.querySelector('.list-view').classList.contains('hidden');

    let videoElements = [];

    if (isListView) {
        // List View: target table rows
        videoElements = Array.from(activeTab.querySelectorAll('.list-view tbody tr'));
        videoElements = videoElements.map(row => {
            // Find the element containing the video type text in the row
            const videoTypeElement = row.querySelector('td:nth-child(4) p');
            const videoType = videoTypeElement ? videoTypeElement.textContent.trim() : '';

            return {
                element: row,
                videoType: videoType,
                status: row.dataset.videoStatus
            };
        });
    } else {
        // Grid View: target video card divs/anchors
        videoElements = Array.from(activeTab.querySelectorAll('.grid-view > [data-video-id]'));
        videoElements = videoElements.map(card => {
            // Fallback for grid view: Look for specific text that indicates video type
            const textNodes = Array.from(card.querySelectorAll('p, h3')).map(el => el.textContent.trim());
            let videoType = '';
            for (const text of textNodes) {
                if (text.includes('Face On') || text.includes('Down The Line')) {
                    videoType = text;
                    break;
                }
            }

            return {
                element: card,
                videoType: videoType,
                status: card.dataset.videoStatus
            };
        });
    }

    // Filter out videos that are not part of the active tab's core content if elements from hidden tabs were somehow selected.
    const container = activeTab.querySelector(isListView ? '.list-view' : '.grid-view');
    const filteredVideos = videoElements.filter(video => container.contains(video.element));

    return { videos: filteredVideos, isListView: isListView };
}

// Function to determine filterable data from videos
function prepareDataSpaceFilterData() {
    const filterKey = 'Video Type';
    // The types as they appear in the HTML after Django templating
    const allPossibleTypes = ['Face On', 'Down The Line', 'Face On & Down The Line'];

    const dataSpaceFilterDefinitions = [{
        name: filterKey,
        originalKey: filterKey,
        values: allPossibleTypes
    }];

    // Get current filters from URL
    const params = new URLSearchParams(window.location.search);
    const activeFilters = new Set(params.getAll('filter_type'));
    const filtersPresent = params.has('filter_type');

    let filterStates = {};
    filterStates[filterKey] = {};

    allPossibleTypes.forEach(val => {
        const normalizedVal = val.toLowerCase().replace('&', '&amp;');
        // If URL has no 'filter_type' params, all are selected (true).
        // If URL has 'filter_type' params, only those in the URL are selected (true).
        if (!filtersPresent) {
            filterStates[filterKey][normalizedVal] = true;
        } else {
            filterStates[filterKey][normalizedVal] = activeFilters.has(val); // Check against the raw value
        }
    });

    return { definitions: dataSpaceFilterDefinitions, states: filterStates };
}

function renderDataSpaceFilterOptions(definitions, filterStates) {
    const container = document.getElementById('table-filter-options-container-data-space');
    if (!container) return;

    container.innerHTML = '';
    // Apply the flex-wrap class to the container element
    container.className = 'flex flex-wrap -mx-2';

    definitions.forEach(col => {
        // Use w-full for single-column filter in Data Space
        const columnSection = document.createElement('div');
        columnSection.className = 'w-full px-2 mb-4 border-b pb-2';

        const safeKey = col.originalKey.replace(/\s+/g, '-');

        // --- LOGIC FOR SELECT ALL ---
        let allValuesChecked = true;
        col.values.forEach(value => {
            const normalizedValue = value.toLowerCase().replace('&', '&amp;');
            if (!(filterStates[col.originalKey]?.[normalizedValue] ?? true)) {
                allValuesChecked = false;
            }
        });

        // Inject the filter title and the "Select All" option
        columnSection.innerHTML = `
            <h3 class="font-semibold mb-2 text-gray-700 md:text-sm text-xs">${col.name}</h3>
            <div class="flex items-center mb-1">
                <input type="checkbox" id="table-select-all-data-space-${safeKey}"
                       class="form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 select-all-checkbox-data-space"
                       data-original-key="${col.originalKey}"
                       ${allValuesChecked ? 'checked' : ''}>
                <label for="table-select-all-data-space-${safeKey}" class="text-gray-700 md:text-sm text-xs cursor-pointer">Select All</label>
            </div>
        `;
        // --- END LOGIC FOR SELECT ALL ---

        col.values.forEach(value => {
            const normalizedValue = value.toLowerCase().replace('&', '&amp;');
            const isChecked = filterStates[col.originalKey]?.[normalizedValue] ?? true;

            const optionDiv = document.createElement('div');
            optionDiv.className = 'flex items-center ml-4 mb-1'; // Added ml-4 for indentation
            const safeValue = value.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '');

            optionDiv.innerHTML = `
                <input type="checkbox" id="table-filter-data-space-${safeKey}-${safeValue}"
                       class="form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 data-space-column-value-checkbox column-data-space-${safeKey}"
                       value="${value}"
                       data-original-key="${col.originalKey}"
                       ${isChecked ? 'checked' : ''}>
                <label for="table-filter-data-space-${safeKey}-${safeValue}" class="text-gray-700 md:text-sm text-xs cursor-pointer">${value}</label>
            `;
            columnSection.appendChild(optionDiv);
        });

        container.appendChild(columnSection);
    });
}


function applyDataSpaceFilters(definitions) {
    const { videos, isListView } = getDataSpaceVideoElements();
    const activeTab = document.querySelector('.tab-content:not(.hidden)');
    if (!activeTab) return;

    // Select containers and elements
    const contentContainer = activeTab.querySelector(isListView ? '.list-view' : '.grid-view');
    const table = isListView ? contentContainer.querySelector('table') : null;
    const filterPanel = document.getElementById('table-filter-panel-data-space');
    if (!filterPanel) return;

    const paginationNav = document.getElementById('pagination-container');
    // The Django-rendered message (which only exists if no videos were initially loaded)
    const initialNoVideosMessage = Array.from(contentContainer.children).find(child =>
        child.tagName === 'P' && child.classList.contains('text-gray-500') && !child.classList.contains('filter-no-videos-message')
    );

    const DATA_SPACE_FILTER_STATE_KEY = 'dataSpaceFilterStates';
    let filterStates = JSON.parse(sessionStorage.getItem(DATA_SPACE_FILTER_STATE_KEY)) || {};

    const filterKey = 'Video Type';

    // 1. DETERMINE/UPDATE STATE
    // If the filter panel is NOT hidden, the user clicked 'Apply'. We must read checkboxes and save the new state.
    if (!filterPanel.classList.contains('hidden')) {
        const checkboxes = filterPanel.querySelectorAll('.data-space-column-value-checkbox');

        checkboxes.forEach(checkbox => {
            const value = checkbox.value.toLowerCase().replace('&', '&amp;');
            // Update the state object
            if (!filterStates[filterKey]) filterStates[filterKey] = {};
            filterStates[filterKey][value] = checkbox.checked;
        });

        sessionStorage.setItem(DATA_SPACE_FILTER_STATE_KEY, JSON.stringify(filterStates));
    }

    // 2. Determine selected types based on the current filterStates
    const selectedVideoTypes = new Set();
    let totalTypes = 0;
    const videoTypeDefinition = definitions.find(d => d.originalKey === filterKey);
    if (videoTypeDefinition) {
        totalTypes = videoTypeDefinition.values.length;
    }

    if (filterStates[filterKey]) {
        Object.keys(filterStates[filterKey]).forEach(key => {
            if (filterStates[filterKey][key] === true) {
                selectedVideoTypes.add(key);
            }
        });
    }

    // 3. Apply filtering and track visible count
    let visibleCount = 0;
    const filterActive = selectedVideoTypes.size > 0 && selectedVideoTypes.size < totalTypes;

    // FIX: Hide non-processing/failed videos if no filter options are selected
    const shouldHideAll = selectedVideoTypes.size === 0 && totalTypes > 0;

    videos.forEach(video => {
        const isProcessingOrFailed = video.status === 'Processing' || video.status === 'Failed';
        let isVisible = true;

        if (shouldHideAll) {
            // Only processing/failed videos are visible when zero filter options are selected.
            isVisible = isProcessingOrFailed;
        } else if (filterActive) {
            // Apply partial filtering
            if (!isProcessingOrFailed) {
                const videoTypeNormalized = video.videoType.toLowerCase().replace('&amp;', '&').trim().replace('&', '&amp;'); // Normalize for state lookup
                isVisible = selectedVideoTypes.has(videoTypeNormalized);
            }
        }
        // else: All videos are visible if all filter types are selected (default state)

        video.element.style.display = isVisible ? '' : 'none';
        if (isVisible) {
            visibleCount++;
        }
    });

    // 4. Handle "No videos found" message display
    let noVideosMessage = contentContainer.querySelector('.filter-no-videos-message');
    if (!noVideosMessage) {
        noVideosMessage = document.createElement('p');
        noVideosMessage.className = `text-center text-gray-500 font-medium filter-no-videos-message hidden ${isListView ? '' : 'col-span-full'}`;
        noVideosMessage.textContent = 'No videos found...';
        // Insert before the first child
        if (contentContainer.firstChild) {
            contentContainer.insertBefore(noVideosMessage, contentContainer.firstChild);
        } else {
            contentContainer.appendChild(noVideosMessage);
        }
    }

    if (visibleCount === 0) {
        // Show the filter message and hide the main content wrappers
        noVideosMessage.classList.remove('hidden');
        if (table) table.classList.add('hidden');
        if (initialNoVideosMessage) initialNoVideosMessage.classList.add('hidden');
        if (paginationNav) paginationNav.classList.add('hidden');
    } else {
        // Hide the filter message and show the main content wrappers
        noVideosMessage.classList.add('hidden');
        if (table) table.classList.remove('hidden');
        if (initialNoVideosMessage) initialNoVideosMessage.classList.add('hidden');
        if (paginationNav) paginationNav.classList.remove('hidden');
    }

    filterPanel.classList.add('hidden');
}

function clearDataSpaceFilters(definitions) {
    const DATA_SPACE_FILTER_STATE_KEY = 'dataSpaceFilterStates';
    let filterStates = {}; // Reset state to all selected for all defined filters

    definitions.forEach(col => {
        filterStates[col.originalKey] = {};
        col.values.forEach(val => {
            filterStates[col.originalKey][val.toLowerCase().replace('&', '&amp;')] = true;
        });
    });
    sessionStorage.setItem(DATA_SPACE_FILTER_STATE_KEY, JSON.stringify(filterStates));

    // Show all rows/cards (and reset hidden content)
    const { videos, isListView } = getDataSpaceVideoElements();
    videos.forEach(video => video.element.style.display = '');

    // Reset pagination and "no videos" message
    const activeTab = document.querySelector('.tab-content:not(.hidden)');
    if (activeTab) {
        const contentContainer = activeTab.querySelector(isListView ? '.list-view' : '.grid-view');
        const table = isListView ? contentContainer.querySelector('table') : null;
        const initialNoVideosMessage = Array.from(contentContainer.children).find(child =>
            child.tagName === 'P' && child.classList.contains('text-gray-500') && !child.classList.contains('filter-no-videos-message')
        );
        const noVideosMessage = contentContainer.querySelector('.filter-no-videos-message');
        const paginationNav = document.getElementById('pagination-container');

        if (table) table.classList.remove('hidden');
        if (initialNoVideosMessage) initialNoVideosMessage.classList.add('hidden');
        if (noVideosMessage) noVideosMessage.classList.add('hidden');
        if (paginationNav) paginationNav.classList.remove('hidden');

        // Re-render to check all boxes and hide panel
        renderDataSpaceFilterOptions(definitions, filterStates);
        const filterPanel = document.getElementById('table-filter-panel-data-space');
        if (filterPanel) filterPanel.classList.add('hidden');
    }
}

function setupDataSpaceEventListeners() {
    const filterDropdownBtn = document.getElementById('table-filter-dropdown-btn-data-space');
    const filterPanel = document.getElementById('table-filter-panel-data-space');
    const applyBtn = document.getElementById('table-apply-filters-btn-data-space');
    const clearBtn = document.getElementById('table-clear-filters-btn-data-space');

    if (!filterDropdownBtn || !filterPanel) return;

    dataSpaceEventHandlers.handleFilterDropdownClick = (event) => {
        event.stopPropagation();
        const isHidden = filterPanel.classList.contains('hidden');
        if (isHidden) {
            // Re-prepare definitions and states *based on URL* every time panel is opened
            const { definitions, states } = prepareDataSpaceFilterData();
            renderDataSpaceFilterOptions(definitions, states);
            filterPanel.classList.remove('hidden');
        } else {
            filterPanel.classList.add('hidden');
        }
    };

    dataSpaceEventHandlers.handleApplyFilters = () => {
        // This is the new apply logic: reload page with URL params
        const params = new URLSearchParams(window.location.search);
        params.delete('filter_type'); // Remove all existing filters
        params.set('page', '1'); // Reset to page 1

        const { definitions } = prepareDataSpaceFilterData(); // Get definitions
        const filterKey = 'Video Type';
        const checkboxes = filterPanel.querySelectorAll('.data-space-column-value-checkbox');
        
        const allValues = new Set();
        const def = definitions.find(d => d.originalKey === filterKey);
        if (def) {
            def.values.forEach(v => allValues.add(v));
        }
        
        const selectedValues = new Set();
        checkboxes.forEach(checkbox => {
            if (checkbox.checked) {
                selectedValues.add(checkbox.value); // 'value' is the raw string e.g. "Face On"
            }
        });

        // Only add filter params if it's a partial selection.
        // If all are selected, it's the same as no filter (no params).
        if (selectedValues.size > 0 && selectedValues.size < allValues.size) {
            selectedValues.forEach(val => {
                params.append('filter_type', val);
            });
        }
        
        // Reload the page
        window.location.search = params.toString();
    };

    dataSpaceEventHandlers.handleClearFilters = () => {
        // This is the new clear logic: reload page, remove filter params
        const params = new URLSearchParams(window.location.search);
        params.delete('filter_type');
        params.set('page', '1'); // Reset to page 1
        window.location.search = params.toString();
    };

    // Delegate event listener for checkboxes (Select All / Value Checkboxes)
    dataSpaceEventHandlers.filterPanelChangeListener = (event) => {
        const key = event.target.dataset.originalKey;
        if (!key) return;

        // Checkbox selectors specific to Data Space
        const valueCheckboxesSelector = `.column-data-space-${key.replace(/\s+/g, '-')}`;
        const selectAllId = `table-select-all-data-space-${key.replace(/\s+/g, '-')}`;

        if (event.target.classList.contains('select-all-checkbox-data-space')) {
            const isChecked = event.target.checked;
            document.querySelectorAll(valueCheckboxesSelector).forEach(cb => cb.checked = isChecked);
        } else if (event.target.classList.contains('data-space-column-value-checkbox')) {
            const allCheckboxes = document.querySelectorAll(valueCheckboxesSelector);
            const allChecked = Array.from(allCheckboxes).every(cb => cb.checked);
            const selectAll = document.getElementById(selectAllId);
            if (selectAll) selectAll.checked = allChecked;
        }
    };

    // Global click handler to close the panel
    dataSpaceEventHandlers.handleDocClick = (event) => {
        if (filterPanel && !filterPanel.classList.contains('hidden') && !filterPanel.contains(event.target) && filterDropdownBtn && !filterDropdownBtn.contains(event.target)) {
            filterPanel.classList.add('hidden');
        }
    };

    // Remove existing document-level listener before re-adding (to prevent multiple instances if called multiple times)
    if(dataSpaceEventHandlers.handleDocClick) document.removeEventListener('click', dataSpaceEventHandlers.handleDocClick);

    // Attach listeners
    filterDropdownBtn.addEventListener('click', dataSpaceEventHandlers.handleFilterDropdownClick);
    if (applyBtn) applyBtn.addEventListener('click', dataSpaceEventHandlers.handleApplyFilters);
    if (clearBtn) clearBtn.addEventListener('click', dataSpaceEventHandlers.handleClearFilters);
    filterPanel.addEventListener('change', dataSpaceEventHandlers.filterPanelChangeListener);
    document.addEventListener('click', dataSpaceEventHandlers.handleDocClick);

    // Initial run
    // NO initial run of applyDataSpaceFilters() needed.
    // The backend handles the initial filtering based on URL params.
}


// --- DOM CONTENT LOADED / PAGE ROUTING LOGIC (Includes Original Results Page logic) ---

document.addEventListener('DOMContentLoaded', function () {
    const isComparePage = document.getElementById('compare-page') !== null;
    // Assuming the Data Space page has an element with this ID
    const isDataSpacePage = document.getElementById('data-space-page') !== null;

    if (isComparePage) {
        // Expose the setup function to the global scope so dashboard_compareSwings.js can call it.
        window.setupComparePageEventListeners = setupComparePageEventListeners;
    } else if (isDataSpacePage) {
        setupDataSpaceEventListeners();
    } else {
        // --- Original Results Page Filtering Logic (Used for the main results page) ---
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
        let filterStates = {}; // Local filterStates for this specific table

        function prepareFilterData() {
            columnFilterDefinitions = [];
            headerNameToVisibleIndexMap = new Map();

            const visibleHeaderTexts = tableHeaders.map(th => th.textContent.trim());

            visibleHeaderTexts.forEach((headerText, visibleColIndex) => {
                headerNameToVisibleIndexMap.set(headerText, visibleColIndex);
            });

            // Define the columns we want to be able to toggle
            const columnsToToggle = [
                'Shoulder Tilt', 'Hip Tilt', 'Shoulder Rotation',
                'Hip Rotation', 'Lead Arm Angle', 'Forward Tilt', 'Knee Bend'
            ];
            
            // Define the column we want to filter by its values
            const columnToFilterByValue = 'Pose Class';

            const tempFilterDefinitionsMap = new Map();

            // 1. Add the "Pose Class" filter definition (value-based)
            const poseClassHeaderText = 'Pose Class';
            const poseClassVisibleIndex = headerNameToVisibleIndexMap.get(poseClassHeaderText);
            
            if (poseClassVisibleIndex !== undefined) {
                const uniqueValues = new Set();
                rows.forEach(row => {
                    const cell = row.querySelectorAll('td')[poseClassVisibleIndex];
                    if (cell) {
                        uniqueValues.add(cell.textContent.trim());
                    }
                });
                let columnValues = Array.from(uniqueValues).filter(val => val !== '');
                columnValues.sort((a, b) => {
                    const numA = parseInt(a.substring(1));
                    const numB = parseInt(b.substring(1));
                    return numA - numB;
                });

                tempFilterDefinitionsMap.set(poseClassHeaderText, {
                    name: poseClassHeaderText,
                    originalKey: poseClassHeaderText,
                    visibleIndex: poseClassVisibleIndex,
                    values: columnValues
                });
            }

            // 2. Add the "Metrics" filter definition (column-toggle-based)
            const metricsKey = 'Metrics';
            // Filter the toggle list to only include columns that *actually exist* in the table
            const availableMetrics = columnsToToggle.filter(colName => headerNameToVisibleIndexMap.has(colName));

            if (availableMetrics.length > 0) {
                tempFilterDefinitionsMap.set(metricsKey, {
                    name: 'Metrics', // This is the filter section title
                    originalKey: metricsKey,
                    visibleIndex: -1, // Not tied to one column
                    values: availableMetrics // The options are the column names
                });
            }

            // 3. Define the order
            const orderedKeys = ['Pose Class', 'Metrics'];

            columnFilterDefinitions = orderedKeys
                .map(key => tempFilterDefinitionsMap.get(key))
                .filter(Boolean); // Filter out undefined if 'Pose Class' or 'Metrics' weren't added

            // 4. Initialize or preserve filter states
            columnFilterDefinitions.forEach(col => {
                if (!filterStates[col.originalKey]) {
                    filterStates[col.originalKey] = {};
                    col.values.forEach(val => {
                        // For both "Pose Class" and "Metrics", default to all selected (true)
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

            // Remove old listeners to prevent duplicates before re-attaching
            document.querySelectorAll('.select-all-checkbox').forEach(checkbox => {
                 checkbox.removeEventListener('change', this.handleSelectAllChange);
                 checkbox.handleSelectAllChange = function() { // Attach handler to element for easy removal
                    const originalKey = this.dataset.originalKey;
                    const isChecked = this.checked;
                    document.querySelectorAll(`.column-${originalKey.replace(/\s+/g, '-')}`).forEach(valCheckbox => {
                        valCheckbox.checked = isChecked;
                    });
                };
                checkbox.addEventListener('change', checkbox.handleSelectAllChange);
            });

            document.querySelectorAll('.column-value-checkbox').forEach(checkbox => {
                 checkbox.removeEventListener('change', this.handleValueChange);
                 checkbox.handleValueChange = function() { // Attach handler to element for easy removal
                    const originalKey = this.dataset.originalKey;
                    const selectAllCheckbox = document.getElementById(`select-all-${originalKey.replace(/\s+/g, '-')}`);
                    const allColumnCheckboxes = document.querySelectorAll(`.column-${originalKey.replace(/\s+/g, '-')}`);
                    const allChecked = Array.from(allColumnCheckboxes).every(cb => cb.checked);

                    if (selectAllCheckbox) selectAllCheckbox.checked = allChecked;
                };
                checkbox.addEventListener('change', checkbox.handleValueChange);
            });
        }

        function applyFilters() {
            const activeFilters = new Map(); // For row-based filters

            // 1. Read all checkbox states from DOM into filterStates
            columnFilterDefinitions.forEach(colDef => {
                filterStates[colDef.originalKey] = {};
                const currentColumnCheckboxes = document.querySelectorAll(`.column-${colDef.originalKey.replace(/\s+/g, '-')}`);
                let selectedValuesCount = 0;
                const selectedValues = new Set();
                
                currentColumnCheckboxes.forEach(checkbox => {
                    const isChecked = checkbox.checked;
                    const value = checkbox.value.toLowerCase();
                    filterStates[colDef.originalKey][value] = isChecked;
                    if (isChecked) {
                        selectedValuesCount++;
                        selectedValues.add(value);
                    }
                });

                // If it's a row-based filter (like Pose Class) and it's partially selected, add to activeFilters
                if (colDef.originalKey === 'Pose Class' && selectedValuesCount > 0 && selectedValuesCount < colDef.values.length) {
                     activeFilters.set(colDef.originalKey, selectedValues);
                }
            });

            // 2. Apply "Metrics" filter (Column Visibility)
            const metricsState = filterStates['Metrics'];
            if (metricsState) {
                tableHeaders.forEach((header, index) => {
                    const headerText = header.textContent.trim();
                    const headerTextLower = headerText.toLowerCase();

                    // Check if this header is one of the metrics we are controlling
                    if (metricsState[headerTextLower] !== undefined) {
                        const showColumn = metricsState[headerTextLower];
                        
                        header.style.display = showColumn ? '' : 'none';
                        rows.forEach(row => {
                            const cell = row.querySelectorAll('td')[index];
                            if (cell) {
                                cell.style.display = showColumn ? '' : 'none';
                            }
                        });
                    }
                });
            }

            // 3. Apply "Pose Class" filter (Row Visibility)
            rows.forEach(row => {
                let rowVisible = true;
                const cells = row.querySelectorAll('td');

                // Only check for "Pose Class"
                const poseClassDef = columnFilterDefinitions.find(c => c.originalKey === 'Pose Class');
                
                if (poseClassDef && activeFilters.has('Pose Class')) {
                    const selectedValues = activeFilters.get('Pose Class');
                    const cellValue = cells[poseClassDef.visibleIndex]?.textContent.trim().toLowerCase();
                    
                    if (!selectedValues.has(cellValue)) {
                        rowVisible = false;
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

            // Ensure all table headers and cells are shown
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
            if (filterPanel && !filterPanel.contains(event.target) && filterDropdownBtn && !filterDropdownBtn.contains(event.target)) {
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