document.addEventListener('DOMContentLoaded', () => {
    const comparePage = document.getElementById('compare-page');
    if (!comparePage) return console.error('No #compare-page');

    //
    // 1) Parse JSON from data-attrs for all three possible sources
    //
    function safeParseDataAttr(raw) {
  if (!raw) return [];
  // \bNaN\b will only match standalone NaN, not part of a string
  const cleaned = raw.replace(/\bNaN\b/g, 'null');
  try {
    return JSON.parse(cleaned);
  } catch (err) {
    console.error('safeParseDataAttr failed:', err, cleaned);
    return [];
  }
}

const allData = {
  v1: safeParseDataAttr(comparePage.getAttribute('data-video1')),
  v2: safeParseDataAttr(comparePage.getAttribute('data-video2')),
  pro: safeParseDataAttr(comparePage.getAttribute('data-video-pro'))
};

    //
    // 2) Grab all URLs & titles
    //
    const allMeta = {
      v1: {
        url:   comparePage.dataset.video1Url,
        title: comparePage.dataset.video1Title
      },
      v2: {
        url:   comparePage.dataset.video2Url,
        title: comparePage.dataset.video2Title
      },
      pro: {
        url:   comparePage.dataset.videoProUrl,
        title: comparePage.dataset.videoProTitle
      }
    };

    //
    // 3) Get video player elements (moved inside updateDatasets for re-selection)
    //
    let vid1 = null; // Declare outside to be accessible globally
    let vid2 = null; // Declare outside to be accessible globally

    //
    // Flags for sync
    //
    let seeking = false;
    let chartSeeking = false;

    function syncTime(sourceVideo, targetVideo) {
        if (!seeking && !chartSeeking) {
            seeking = true;
            targetVideo.currentTime = sourceVideo.currentTime;
            setTimeout(() => seeking = false, 100);
        }
    }

    // Event listeners will be added dynamically when video players are updated

    //
    // 4) Table & badge helpers (unchanged)
    //
    function makeCell(val, cls = []) {
        const td = document.createElement('td');
        td.textContent = (typeof val === 'number') ? val.toFixed(2)
            : (val == null || val === '') ? '-' : val;
        td.classList.add(...cls, 'font-medium', 'px-6', 'py-4', 'whitespace-nowrap', 'text-sm', 'text-gray-700');
        return td;
    }

    // MODIFIED: makeBadge function to include 'Very Bad' status and correct colors
    function makeBadge(status) {
        const div = document.createElement('div');
        div.classList.add('inline-flex','items-center','px-3','py-1','text-xs','font-medium','rounded-full');
        if (status === 'Good')    div.classList.add('bg-green-100','text-green-800');
        else if (status === 'Neutral') div.classList.add('bg-gray-200','text-gray-800');
        else if (status === 'Bad') div.classList.add('bg-yellow-100','text-yellow-800'); // Changed to yellow
        else if (status === 'Very Bad') div.classList.add('bg-red-100','text-red-800'); // Added Very Bad
        else div.classList.add('bg-gray-200','text-gray-800'), status='-';
        div.textContent = status;
        const td = document.createElement('td');
        td.classList.add('px-6','py-4','whitespace-nowrap');
        td.appendChild(div);
        return td;
    }

    // MODIFIED: getStatus function to include 'Very Bad'
    function getStatus(diff) {
        const VERY_BAD_THRESHOLD = -5; // Example threshold, adjust as needed based on your data
        if (diff > 0) return 'Good';
        if (diff < VERY_BAD_THRESHOLD) return 'Very Bad';
        if (diff < 0) return 'Bad'; // This will catch diff between VERY_BAD_THRESHOLD and 0
        if (diff === 0) return 'Neutral';
        return '-';
    }

    //
    // 5) Core Pose & Tilt definitions
    //
    const poseClasses = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10'];
    const tiltTypes   = ['Shoulder Tilt','Hip Tilt'];

    //
    // State: which two data sets are we comparing?
    //
    let currentMode = 'v1v2'; // 'v1v2' | 'prov1' | 'prov2'

    //
    // Containers
    //
    const tbody      = document.getElementById('combined-data-body'); // This needs to be dynamically re-selected or updated based on which panel is visible.
    const tableView  = document.getElementById('table-view'); // This needs to be dynamically re-selected or updated based on which panel is visible.
    const chartView  = document.getElementById('chart-view'); // This needs to be dynamically re-selected or updated based on which panel is visible.
    const tableTab   = document.getElementById('table-tab'); // This needs to be dynamically re-selected or updated based on which panel is visible.
    const chartTab   = document.getElementById('chart-tab'); // This needs to be dynamically re-selected or updated based on which panel is visible.
    let chartInstance= null;

    //
    // 6) Switch out which data, titles & URLs we use
    //
    function updateDatasets() {
      let srcA, srcB, mA, mB;
      if (currentMode === 'v1v2') {
        srcA = allData.v1; srcB = allData.v2;
        mA   = allMeta.v1; mB   = allMeta.v2;
      } else if (currentMode === 'prov1') {
        srcA = allData.pro; srcB = allData.v1;
        mA   = allMeta.pro; mB   = allMeta.v1;
      } else {
        srcA = allData.pro; srcB = allData.v2;
        mA   = allMeta.pro; mB   = allMeta.v2;
      }

      // Update video sources & titles. These are now global for the chart labels.
      window.video1Title = mA.title;
      window.video2Title = mB.title;

      // Select video player elements for the currently active panel
      // It's crucial to select them AFTER the panel content has been rendered.
      vid1 = document.getElementById('video-player-left');
      vid2 = document.getElementById('video-player-right');

      if (!vid1 || !vid2) {
          console.error('Missing <video> tags (left or right player not found) in current panel.');
          return; // Exit if elements not found in current panel
      }

      // Remove existing event listeners to prevent duplicates
      ['seeking', 'seeked'].forEach(evt => {
          vid1.removeEventListener(evt, () => syncTime(vid1, vid2));
          vid2.removeEventListener(evt, () => syncTime(vid2, vid1));
      });

      // Add new event listeners
      ['seeking', 'seeked'].forEach(evt => {
          vid1.addEventListener(evt, () => syncTime(vid1, vid2));
          vid2.addEventListener(evt, () => syncTime(vid2, vid1));
      });

      vid1.src = mA.url;
      vid2.src = mB.url;
      vid1.load(); // Load the new source
      vid2.load(); // Load the new source

      // reassign JSON maps
      map1 = new Map(srcA.map(r => [r['Pose Class'], r]));
      map2 = new Map(srcB.map(r => [r['Pose Class'], r]));
    }

    //
    // 7) Build combined table
    //
    let map1 = new Map(), map2 = new Map();
    function buildTable() {
      // Re-select tbody for the currently active panel
      const currentTbody = document.querySelector('.tab-content:not(.hidden) #combined-data-body');
      if (!currentTbody) {
          console.error('Table body not found in the active tab content.');
          return;
      }
      currentTbody.innerHTML = '';
      if (!map1.size && !map2.size) {
        const tr = document.createElement('tr');
        const td = makeCell('No data to compare.', ['text-center']);
        td.colSpan = 6;
        tr.appendChild(td);
        currentTbody.appendChild(tr);
        return;
      }
      poseClasses.forEach(pose => {
        tiltTypes.forEach(metric => {
          const tr = document.createElement('tr');
          tr.classList.add('border-b','hover:bg-indigo-200','cursor-pointer');

          const t1 = map1.get(pose)?.['Time Frame'] ?? null;
          const t2 = map2.get(pose)?.['Time Frame'] ?? null;
          if (t1!=null) tr.dataset.timestampV1 = t1;
          if (t2!=null) tr.dataset.timestampV2 = t2;

          tr.appendChild(makeCell(pose,   ['font-medium','text-gray-900']));
          tr.appendChild(makeCell(metric, ['font-medium','text-gray-900']));

          const v1 = parseFloat(map1.get(pose)?.[metric]);
          const v2 = parseFloat(map2.get(pose)?.[metric]);
          tr.appendChild(makeCell(isNaN(v1)?null:v1));
          tr.appendChild(makeCell(isNaN(v2)?null:v2));

          let diff = null;
          if (!isNaN(v1) && !isNaN(v2)) diff = +(v2 - v1).toFixed(2);
          tr.appendChild(makeCell(diff));

          const status = diff==null?'-':getStatus(diff);
          tr.appendChild(makeBadge(status));

          // MODIFIED: Apply row background based on status
          if (status === 'Good') {
            tr.classList.add('bg-green-100');
          } else if (status === 'Bad') {
            tr.classList.add('bg-yellow-100');
          } else if (status === 'Very Bad') {
            tr.classList.add('bg-red-100');
          } else {
            tr.classList.add('bg-white'); // Default for Neutral or other cases
          }

          tr.addEventListener('click',()=>{
            const ts1 = parseFloat(tr.dataset.timestampV1);
            const ts2 = parseFloat(tr.dataset.timestampV2);
            // Ensure vid1 and vid2 are currently selected elements
            const currentVid1 = document.getElementById('video-player-left');
            const currentVid2 = document.getElementById('video-player-right');

            if (!isNaN(ts1) && currentVid1) { currentVid1.currentTime = ts1; currentVid1.pause(); }
            if (!isNaN(ts2) && currentVid2) { currentVid2.currentTime = ts2; currentVid2.pause(); }
          });

          currentTbody.appendChild(tr);
        });
      });
    }

    //
    // 8) Build the line chart
    //
    function buildChart() {
      // Re-select chart canvas for the currently active panel
      const currentChartCanvas = document.querySelector('.tab-content:not(.hidden) #tiltChart');
      if (!currentChartCanvas) {
          console.error('Chart canvas not found in the active tab content.');
          return;
      }

      const hipA  = poseClasses.map(p=>parseFloat(map1.get(p)?.['Hip Tilt']));
      const hipB  = poseClasses.map(p=>parseFloat(map2.get(p)?.['Hip Tilt']));
      const shA   = poseClasses.map(p=>parseFloat(map1.get(p)?.['Shoulder Tilt']));
      const shB   = poseClasses.map(p=>parseFloat(map2.get(p)?.['Shoulder Tilt']));

      if (chartInstance) chartInstance.destroy();
      const ctx = currentChartCanvas.getContext('2d');
      chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: poseClasses,
          datasets: [
            { label: `${window.video1Title} - Shoulder Tilt`, data: shA, fill:false, tension:0.2 },
            { label: `${window.video2Title} - Shoulder Tilt`, data: shB, fill:false, tension:0.2 },
            { label: `${window.video1Title} - Hip Tilt`,      data: hipA, fill:false, tension:0.2 },
            { label: `${window.video2Title} - Hip Tilt`,      data: hipB, fill:false, tension:0.2 }
          ]
        },
        options: {
          responsive:true,
          scales:{
            y:{title:{display:true,text:'Tilt Angle'}},
            x:{title:{display:true,text:'Pose Class'}}
          },
          onClick:(evt, elems) =>{
            if (!elems.length) return;
            chartSeeking = true;
            const {datasetIndex,index} = elems[0];
            const pose = poseClasses[index];
            let ts;
            // Determine which map to use based on datasetIndex
            if (datasetIndex === 0 || datasetIndex === 2) { // Shoulder Tilt (Video 1) or Hip Tilt (Video 1)
                ts = map1.get(pose)?.['Time Frame'];
            } else { // Shoulder Tilt (Video 2) or Hip Tilt (Video 2)
                ts = map2.get(pose)?.['Time Frame'];
            }

            // Ensure vid1 and vid2 are currently selected elements
            const currentVid1 = document.getElementById('video-player-left');
            const currentVid2 = document.getElementById('video-player-right');

            const vid = (datasetIndex === 0 || datasetIndex === 2) ? currentVid1 : currentVid2;

            if (vid && ts) {
                vid.currentTime = parseFloat(ts);
                vid.pause();
            }
            setTimeout(()=>chartSeeking=false,200);
          },
          plugins:{ tooltip:{mode:'index',intersect:false} }
        }
      });
    }

    //
    // 9) Inner “Table vs Chart” toggles
    // Moved inside a function to be called after panel update
    //
    function setupInnerTabs() {
      // Re-select tabs and views for the currently active panel
      const currentTableView = document.querySelector('.tab-content:not(.hidden) #table-view');
      const currentChartView = document.querySelector('.tab-content:not(.hidden) #chart-view');
      const currentTableTab = document.querySelector('.tab-content:not(.hidden) #table-tab');
      const currentChartTab = document.querySelector('.tab-content:not(.hidden) #chart-tab');

      if (!currentTableView || !currentChartView || !currentTableTab || !currentChartTab) {
          console.error('Inner tabs or views not found in the active tab content.');
          return;
      }

      // Define the handler functions.
      // These must be defined before they are used in removeEventListener.
      const handleTableTabClick = () => {
        currentTableView.classList.remove('hidden');
        currentChartView.classList.add('hidden');
        currentTableTab.classList.replace('text-gray-500','text-blue-600');
        currentTableTab.classList.replace('border-transparent','border-blue-600');
        currentChartTab.classList.replace('text-blue-600','text-gray-500');
        currentChartTab.classList.replace('border-blue-600','border-transparent');
      };

      const handleChartTabClick = () => {
        currentChartView.classList.remove('hidden');
        currentTableView.classList.add('hidden');
        currentChartTab.classList.replace('text-gray-500','text-blue-600');
        currentChartTab.classList.replace('border-transparent','border-blue-600');
        currentTableTab.classList.replace('text-blue-600','text-gray-500');
        currentTableTab.classList.replace('border-blue-600','border-transparent');
      };

      // Remove existing event listeners to prevent duplicates.
      // Now, handleTableTabClick and handleChartTabClick are defined.
      currentTableTab.removeEventListener('click', handleTableTabClick);
      currentChartTab.removeEventListener('click', handleChartTabClick);

      currentTableTab.addEventListener('click', handleTableTabClick);
      currentChartTab.addEventListener('click', handleChartTabClick);

      // Default to table view
      currentTableTab.click();
    }


    //
    // 10) Outer tabs: switch currentMode
    //
    document.querySelectorAll('[data-tab]').forEach(btn => {
      btn.addEventListener('click', () => {
        // style all, then highlight this one
        document.querySelectorAll('[data-tab]').forEach(b=>{
          b.classList.replace('text-blue-600','text-gray-500');
          b.classList.replace('border-blue-600','border-transparent');
        });
        btn.classList.replace('text-gray-500','text-blue-600');
        btn.classList.replace('border-transparent','border-blue-600');

        // show/hide the three panels
        currentMode = btn.dataset.tab; // 'v1v2' / 'prov1' / 'prov2'
        document.querySelectorAll('.tab-content').forEach(p=>p.classList.add('hidden'));
        document.getElementById(`tab-${currentMode}`).classList.remove('hidden');

        // Reassign data & rebuild
        // IMPORTANT: Call updateDatasets *before* building table/chart
        updateDatasets();
        // Now, setup the inner tabs for the newly visible panel
        setupInnerTabs(); // This will also click the table tab by default
        buildTable();
        const chartPane = document.querySelector('.tab-content:not(.hidden) #chart-view');
chartPane.classList.remove('hidden');

// 2. Build the chart now that the canvas has dimensions
buildChart();

// 3. Re-hide the chart pane so the user still sees the table by default
chartPane.classList.add('hidden');
      });
    });

    //
    // 11) Initialize first view
    //
    updateDatasets();
    setupInnerTabs(); // Call this initially to set up the default v1v2 panel
    buildTable();
    const chartPane = document.querySelector('.tab-content:not(.hidden) #chart-view');
chartPane.classList.remove('hidden');

// 2. Build the chart now that the canvas has dimensions
buildChart();

// 3. Re-hide the chart pane so the user still sees the table by default
chartPane.classList.add('hidden');
});