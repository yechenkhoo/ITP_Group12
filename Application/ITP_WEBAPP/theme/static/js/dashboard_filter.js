// A global container for our named event handlers to allow for easy removal.
const comparePageEventHandlers = {};

// Hoist these variables to a higher scope so their state persists across function calls
// These are primarily used by the Compare Page logic.
let tableColumnFilterDefinitions = [];
let tableFilterStates = {}; // This will persist across tab switches

// --- NEW DATA SPACE FILTERING LOGIC ---

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
function prepareDataSpaceFilterData(videos) {
    const filterKey = 'Video Type';
    // The types as they appear in the HTML after Django templating
    const allPossibleTypes = ['Face On', 'Down The Line', 'Face On & Down The Line'];
    const uniqueValues = new Set();
    
    videos.filter(v => v.videoType && v.status !== 'Processing' && v.status !== 'Failed').forEach(v => {
         // Normalize and check against possible types
         const type = v.videoType.replace('&amp;', '&').trim();
         if (allPossibleTypes.includes(type)) {
             uniqueValues.add(type);
         }
    });

    let columnValues = Array.from(uniqueValues).filter(val => val !== '');
    
    // Maintain the specific order for the display
    columnValues = allPossibleTypes.filter(value => columnValues.includes(value));
    
    const dataSpaceFilterDefinitions = [];
    if (columnValues.length > 0) {
        dataSpaceFilterDefinitions.push({
            name: filterKey,
            originalKey: filterKey,
            values: columnValues
        });
    }

    const DATA_SPACE_FILTER_STATE_KEY = 'dataSpaceFilterStates';
    let filterStates = JSON.parse(sessionStorage.getItem(DATA_SPACE_FILTER_STATE_KEY)) || {};

    // Initialize/sync filter states
    dataSpaceFilterDefinitions.forEach(col => {
        if (!filterStates[col.originalKey]) {
            filterStates[col.originalKey] = {};
        }
        col.values.forEach(val => {
            const lowerCaseVal = val.toLowerCase();
            if (filterStates[col.originalKey][lowerCaseVal] === undefined) {
                filterStates[col.originalKey][lowerCaseVal] = true; // Default to selected
            }
        });
    });
    
    // Clean up states for types that no longer exist in the data
    if (filterStates[filterKey]) {
        Object.keys(filterStates[filterKey]).forEach(key => {
            if (!columnValues.map(v => v.toLowerCase()).includes(key)) {
                delete filterStates[filterKey][key];
            }
        });
    }
    
    sessionStorage.setItem(DATA_SPACE_FILTER_STATE_KEY, JSON.stringify(filterStates));

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
        
        // --- NEW LOGIC FOR SELECT ALL ---
        let allValuesChecked = true;
        col.values.forEach(value => {
            if (!(filterStates[col.originalKey]?.[value.toLowerCase()] ?? true)) {
                allValuesChecked = false;
            }
        });

        // Inject the filter title and the new "Select All" option
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
        // --- END NEW LOGIC FOR SELECT ALL ---
        
        col.values.forEach(value => {
            const isChecked = filterStates[col.originalKey]?.[value.toLowerCase()] ?? true;
            
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

    // --- MODIFICATION START: Select Pagination Container using its new ID ---
    const paginationNav = document.getElementById('pagination-container');
    // --- MODIFICATION END: Select Pagination Container ---

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
            const value = checkbox.value.toLowerCase();
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
                const videoTypeNormalized = video.videoType.toLowerCase().replace('&amp;', '&').trim();
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
        
        // --- MODIFICATION: Hide Pagination ---
        if (paginationNav) paginationNav.classList.add('hidden');
        // --- MODIFICATION: Hide Pagination ---
        
    } else {
        // Hide the filter message and show the main content wrappers
        noVideosMessage.classList.add('hidden');
        if (table) table.classList.remove('hidden');
        if (initialNoVideosMessage) initialNoVideosMessage.classList.add('hidden'); 

        // --- MODIFICATION: Show Pagination ---
        if (paginationNav) paginationNav.classList.remove('hidden');
        // --- MODIFICATION: Show Pagination ---
    }
    
    filterPanel.classList.add('hidden');
}


function clearDataSpaceFilters(definitions) {
    const DATA_SPACE_FILTER_STATE_KEY = 'dataSpaceFilterStates';
    let filterStates = {};

    // Reset state to all selected for all defined filters
    definitions.forEach(col => {
        filterStates[col.originalKey] = {};
        col.values.forEach(val => {
            filterStates[col.originalKey][val.toLowerCase()] = true;
        });
    });
    
    sessionStorage.setItem(DATA_SPACE_FILTER_STATE_KEY, JSON.stringify(filterStates));

    // Show all rows/cards
    const { videos, isListView } = getDataSpaceVideoElements();
    videos.forEach(video => video.element.style.display = '');

    // Re-render to check all boxes and hide panel
    const filterPanel = document.getElementById('table-filter-panel-data-space');
    
    const activeTab = document.querySelector('.tab-content:not(.hidden)');
    if (activeTab) {
        const contentContainer = activeTab.querySelector(isListView ? '.list-view' : '.grid-view');
        const table = isListView ? contentContainer.querySelector('table') : null;
        
        // --- MODIFICATION START: Select Pagination Container using its new ID ---
        const paginationNav = document.getElementById('pagination-container');
        // --- MODIFICATION END: Select Pagination Container ---
        
        // Find initial message using robust selector
        const initialNoVideosMessage = Array.from(contentContainer.children).find(child => 
            child.tagName === 'P' && child.classList.contains('text-gray-500') && !child.classList.contains('filter-no-videos-message')
        );

        // Hide the filter 'No videos' message
        contentContainer.querySelector('.filter-no-videos-message')?.classList.add('hidden');
        
        // If there are videos, show the table/grid and hide the initial message. If not, show the initial message.
        if (videos.length > 0) {
             if (table) table.classList.remove('hidden');
             if (initialNoVideosMessage) initialNoVideosMessage.classList.add('hidden');

             // --- MODIFICATION: Show Pagination ---
             if (paginationNav) paginationNav.classList.remove('hidden');
             // --- MODIFICATION: Show Pagination ---

        } else {
             // In this case, only the initial message should be visible if it was rendered by Django
             if (table) table.classList.add('hidden'); 
             if (initialNoVideosMessage) initialNoVideosMessage.classList.remove('hidden');

             // --- MODIFICATION: Hide Pagination ---
             if (paginationNav) paginationNav.classList.add('hidden');
             // --- MODIFICATION: Hide Pagination ---
        }
    }
    
    if (filterPanel) { 
        renderDataSpaceFilterOptions(definitions, filterStates); // Re-render to check all boxes
        filterPanel.classList.add('hidden');
    }
}


function setupDataSpaceEventListeners() {
    // NOTE: IDs are updated to match the HTML
    const filterDropdownBtn = document.getElementById('table-filter-dropdown-btn-data-space');
    const applyFiltersBtn = document.getElementById('table-apply-filters-btn-data-space');
    const clearFiltersBtn = document.getElementById('table-clear-filters-btn-data-space');
    const filterPanel = document.getElementById('table-filter-panel-data-space');

    if (!filterDropdownBtn || !filterPanel) return; 

    let currentDataSpaceDefinitions = [];

    const handleDropdownClick = function(event) {
        event.stopPropagation();
        
        const { videos } = getDataSpaceVideoElements();
        const { definitions, states } = prepareDataSpaceFilterData(videos);
        currentDataSpaceDefinitions = definitions;

        if (currentDataSpaceDefinitions.length === 0) {
            filterPanel.classList.add('hidden');
            return;
        }
        
        renderDataSpaceFilterOptions(currentDataSpaceDefinitions, states);
        filterPanel.classList.toggle('hidden');
    };
    
    const handleDocClick = function(event) {
        if (!filterPanel.classList.contains('hidden')) {
            if (!filterPanel.contains(event.target) && !filterDropdownBtn.contains(event.target)) {
                filterPanel.classList.add('hidden');
            }
        }
    };
    
    const handleApplyClick = () => {
        applyDataSpaceFilters(currentDataSpaceDefinitions);
    };
    
    const handleClearClick = () => {
        clearDataSpaceFilters(currentDataSpaceDefinitions);
    };

    filterDropdownBtn.addEventListener('click', handleDropdownClick);
    document.addEventListener('click', handleDocClick);
    filterPanel.addEventListener('click', function(event) { event.stopPropagation(); });
    applyFiltersBtn.addEventListener('click', handleApplyClick);
    clearFiltersBtn.addEventListener('click', handleClearClick);
    
    // --- NEW: Add event delegation for "Select All" and individual checkboxes ---
    filterPanel.addEventListener('change', (event) => {
        const key = event.target.dataset.originalKey;
        const safeKey = key ? key.replace(/\s+/g, '-') : null;

        if (event.target.classList.contains('select-all-checkbox-data-space')) {
            // Handle "Select All" click
            const isChecked = event.target.checked;
            filterPanel.querySelectorAll(`.column-data-space-${safeKey}`).forEach(cb => cb.checked = isChecked);
        } else if (event.target.classList.contains('data-space-column-value-checkbox')) {
            // Handle individual checkbox click
            const allCheckboxes = filterPanel.querySelectorAll(`.column-data-space-${safeKey}`);
            const allChecked = Array.from(allCheckboxes).every(cb => cb.checked);
            const selectAll = filterPanel.querySelector(`#table-select-all-data-space-${safeKey}`);
            if(selectAll) selectAll.checked = allChecked;
        }
    });
    // --- END NEW EVENT LISTENER ---
    
    // Initial load logic: setup definitions and apply existing filter state
    const { videos } = getDataSpaceVideoElements();
    if (videos.length > 0) {
        const { definitions } = prepareDataSpaceFilterData(videos);
        currentDataSpaceDefinitions = definitions;
        // This is the call that needed the fix in applyDataSpaceFilters to rely on session state
        applyDataSpaceFilters(currentDataSpaceDefinitions); 
    }
}
// --- END OF NEW DATA SPACE FILTERING LOGIC ---

function setupComparePageEventListeners() {
    // --- Helper function to get the currently active tab panel ---
    const getActivePanel = () => document.querySelector('.tab-content:not(.hidden)');

    // --- Helper function to get elements from the active tab ---\
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
            
            let allValuesChecked = true;
            col.values.forEach(value => {
                const isChecked = tableFilterStates[col.originalKey]?.[value.toLowerCase()] ?? true;
                if (!isChecked) allValuesChecked = false;
            });

            columnSection.innerHTML = `
                <h3 class="font-semibold mb-2 text-gray-700 md:text-sm text-xs">${col.name}</h3>
                <div class="flex items-center mb-1">
                    <input type="checkbox" id="table-select-all-${safeKey}" class="form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 select-all-checkbox" data-original-key="${col.originalKey}" ${allValuesChecked ? 'checked' : ''}>
                    <label for="table-select-all-${safeKey}" class="text-gray-700 md:text-sm text-xs cursor-pointer">Select All</label>
                </div>
            `;
            
            col.values.forEach(value => {
                const isChecked = tableFilterStates[col.originalKey]?.[value.toLowerCase()] ?? true;
                
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
    const isDataSpacePage = document.getElementById('table-filter-dropdown-btn-data-space') !== null;

    if (isComparePage) {
        // Expose the setup function to the global scope so dashboard_compareSwings.js can call it.
        window.setupComparePageEventListeners = setupComparePageEventListeners;
    } else if (isDataSpacePage) {
        // --- Data Space Page Filtering Logic ---
        setupDataSpaceEventListeners();
    } else {
        // --- Original Results Page Filtering Logic ---
        // This logic handles the original results.html filtering, preserved here for completeness.
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
                'Time Frame',
                'Shoulder Tilt',
                'Hip Tilt',
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
                    name: headerText, originalKey: originalColKey, visibleIndex: visibleColIndex, values: columnValues
                });
            });

            // Re-order based on expected UI
            const orderedKeys = ['Time Frame', 'Overall Status', 'Pose Class', 'Body Tilt', 'Improvement Status'];
            columnFilterDefinitions = orderedKeys
                .map(key => tempFilterDefinitionsMap.get(key))
                .filter(Boolean); // Filter out any undefined entries

            // Initialize filter states (default to selected)
            columnFilterDefinitions.forEach(col => {
                if (!filterStates[col.originalKey]) {
                    filterStates[col.originalKey] = {};
                    col.values.forEach(val => {
                        filterStates[col.originalKey][val.toLowerCase()] = true; // Default to selected
                    });
                }
            });

            // Ensure Body Tilt filter is created if Shoulder/Hip Tilt columns exist but are excluded from the main loop
            if (!filterStates['Body Tilt'] && (headerNameToVisibleIndexMap.has('Shoulder Tilt') || headerNameToVisibleIndexMap.has('Hip Tilt'))) {
                columnFilterDefinitions.push({
                    name: 'Body Tilt',
                    originalKey: 'Body Tilt',
                    values: ['Shoulder Tilt', 'Hip Tilt']
                });
                filterStates['Body Tilt'] = { 'shoulder tilt': true, 'hip tilt': true };
            }
        }

        function renderFilterOptions() {
            filterOptionsContainer.innerHTML = '';
            filterOptionsContainer.className = 'flex flex-wrap -mx-2';

            columnFilterDefinitions.forEach(col => {
                const columnSection = document.createElement('div');
                columnSection.className = 'w-1/2 px-2 mb-4 border-b pb-2';

                // Sanitize key for use in IDs
                const safeKey = col.originalKey.replace(/\s+/g, '-');

                let allValuesChecked = true;
                col.values.forEach(value => {
                    const isChecked = filterStates[col.originalKey]?.[value.toLowerCase()] ?? true;
                    if (!isChecked) allValuesChecked = false;
                });

                columnSection.innerHTML = `
                    <h3 class="font-semibold mb-2 text-gray-700 md:text-sm text-xs">${col.name}</h3>
                    <div class="flex items-center mb-1">
                        <input type="checkbox" id="filter-select-all-${safeKey}" class="form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 select-all-checkbox" data-original-key="${col.originalKey}" ${allValuesChecked ? 'checked' : ''}>
                        <label for="filter-select-all-${safeKey}" class="text-gray-700 md:text-sm text-xs cursor-pointer">Select All</label>
                    </div>
                `;

                col.values.forEach(value => {
                    const isChecked = filterStates[col.originalKey]?.[value.toLowerCase()] ?? true;

                    const optionDiv = document.createElement('div');
                    optionDiv.className = 'flex items-center ml-4 mb-1';
                    // Sanitize value for use in IDs
                    const safeValue = value.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '');

                    optionDiv.innerHTML = `
                        <input type="checkbox" id="filter-${safeKey}-${safeValue}" class="form-checkbox h-4 w-4 text-indigo-600 rounded mr-2 column-value-checkbox column-${safeKey}" value="${value}" data-original-key="${col.originalKey}" ${isChecked ? 'checked' : ''}>
                        <label for="filter-${safeKey}-${safeValue}" class="text-gray-700 md:text-sm text-xs cursor-pointer">${value}</label>
                    `;
                    columnSection.appendChild(optionDiv);
                });
                
                filterOptionsContainer.appendChild(columnSection);
            });

            // Event listener for select-all checkboxes
            filterOptionsContainer.querySelectorAll('.select-all-checkbox').forEach(selectAll => {
                selectAll.addEventListener('change', (event) => {
                    const key = event.target.dataset.originalKey;
                    const isChecked = event.target.checked;
                    filterOptionsContainer.querySelectorAll(`.column-${key.replace(/\s+/g, '-')}`).forEach(cb => cb.checked = isChecked);
                });
            });

            // Event listener for individual value checkboxes
            filterOptionsContainer.querySelectorAll('.column-value-checkbox').forEach(checkbox => {
                checkbox.addEventListener('change', (event) => {
                    const key = event.target.dataset.originalKey;
                    const allCheckboxes = filterOptionsContainer.querySelectorAll(`.column-${key.replace(/\s+/g, '-')}`);
                    const allChecked = Array.from(allCheckboxes).every(cb => cb.checked);
                    const selectAll = filterOptionsContainer.querySelector(`#filter-select-all-${key.replace(/\s+/g, '-')}`);
                    if(selectAll) selectAll.checked = allChecked;
                });
            });
        }

        function applyFilters() {
            const activeFilters = new Map();

            columnFilterDefinitions.forEach(colDef => {
                const checkboxes = filterOptionsContainer.querySelectorAll(`.column-${colDef.originalKey.replace(/\s+/g, '-')}`);
                filterStates[colDef.originalKey] = {};
                const selectedValues = new Set();

                checkboxes.forEach(checkbox => {
                    const isChecked = checkbox.checked;
                    filterStates[colDef.originalKey][checkbox.value.toLowerCase()] = isChecked;
                    if (isChecked) selectedValues.add(checkbox.value.toLowerCase());
                });

                // Only add to activeFilters if it's a partial selection
                if (selectedValues.size > 0 && selectedValues.size < colDef.values.length) {
                    activeFilters.set(colDef.originalKey, selectedValues);
                }
            });

            rows.forEach(row => {
                let rowVisible = true;
                const cells = row.querySelectorAll('td');

                for (const colDef of columnFilterDefinitions) {
                    if (activeFilters.has(colDef.originalKey)) {
                        let cellValue;

                        if (colDef.originalKey === 'Body Tilt') {
                            const shoulderTiltIndex = headerNameToVisibleIndexMap.get('Shoulder Tilt');
                            const hipTiltIndex = headerNameToVisibleIndexMap.get('Hip Tilt');
                            const shoulderTiltValue = cells[shoulderTiltIndex]?.textContent.trim().toLowerCase();
                            const hipTiltValue = cells[hipTiltIndex]?.textContent.trim().toLowerCase();

                            // The row is visible if at least one selected tilt type has a value in the cell
                            let tiltVisible = false;
                            if (activeFilters.get(colDef.originalKey).has('shoulder tilt') && shoulderTiltValue) {
                                tiltVisible = true;
                            }
                            if (activeFilters.get(colDef.originalKey).has('hip tilt') && hipTiltValue) {
                                tiltVisible = true;
                            }

                            if (!tiltVisible) {
                                rowVisible = false;
                                break;
                            }
                        } else {
                            const visibleIndex = colDef.visibleIndex;
                            cellValue = cells[visibleIndex]?.textContent.trim().toLowerCase();

                            if (!activeFilters.get(colDef.originalKey).has(cellValue)) {
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
                if (filterStates[col.originalKey]) {
                    col.values.forEach(val => {
                        filterStates[col.originalKey][val.toLowerCase()] = true;
                    });
                }
            });

            // Set filters to checked
            filterOptionsContainer.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = true;
            });

            // Show all rows
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