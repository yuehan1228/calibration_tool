import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { PCDLoader } from './vendor/PCDLoader.js';

const FRAME_PALETTE = [
    0x38bdf8, 0xfbbf24, 0xa78bfa, 0x34d399,
    0xf472b6, 0xfb7185, 0x60a5fa, 0xfacc15
];

const state = {
    frames: [],       // { id, name, visible, colorHex, pointCount, hasIntensity, pointsObj, rawPositions, rawIntensity, bbox }
    nextFrameId: 1,
    calibPoints: [],  // { id, frameName, radar: [x,y,z], blh: [B,L,H], enabled, status: 'pending'|'kept'|'removed', error, removedRound }
    nextPointId: 1,
    pickMode: false,
    shiftDown: false,
    colorMode: 'height',
    pointSize: 2.5,
    pickThreshold: 0.25,
    markerScale: 0.2,
};

// ===== DOM 引用 =====
const dom = {
    fileInput: document.getElementById('pcd_file_input'),
    frameList: document.getElementById('pcd_frame_list'),
    modeViewBtn: document.getElementById('pcd_mode_view_btn'),
    modePickBtn: document.getElementById('pcd_mode_pick_btn'),
    colorMode: document.getElementById('pcd_color_mode'),
    pointSize: document.getElementById('pcd_point_size'),
    pickThreshold: document.getElementById('pcd_pick_threshold'),
    pickThresholdVal: document.getElementById('pcd_pick_threshold_val'),
    resetViewBtn: document.getElementById('pcd_reset_view_btn'),
    topViewBtn: document.getElementById('pcd_top_view_btn'),
    clearFramesBtn: document.getElementById('pcd_clear_frames_btn'),
    viewerWrap: document.getElementById('pcd_viewer_wrap'),
    canvasContainer: document.getElementById('pcd_canvas_container'),
    hud: document.getElementById('pcd_hud'),
    addManualBtn: document.getElementById('pcd_add_manual_btn'),
    exportJsonBtn: document.getElementById('pcd_export_json_btn'),
    importJsonBtn: document.getElementById('pcd_import_json_btn'),
    importJsonInput: document.getElementById('pcd_import_json_input'),
    clearPointsBtn: document.getElementById('pcd_clear_points_btn'),
    pointsTbody: document.getElementById('pcd_points_tbody'),
    pointsCountHint: document.getElementById('pcd_points_count_hint'),
    coordSystem: document.getElementById('pcd_coord_system'),
    armParams: document.getElementById('pcd_arm_params'),
    connectParams: document.getElementById('pcd_connect_params'),
    calcBtn: document.getElementById('pcd_calculate_btn'),
    errorBox: document.getElementById('pcd_error_message'),
    resultSection: document.getElementById('pcd_result_section'),
    calibBanner: document.getElementById('pcd_calib_banner'),
    matrixResult: document.getElementById('pcd_matrix_result'),
    matrixNumpy: document.getElementById('pcd_matrix_numpy'),
    copyMatrixBtn: document.getElementById('pcd_copy_matrix_btn'),
    iterTbody: document.getElementById('pcd_iter_tbody'),
    errorSummary: document.getElementById('pcd_error_summary'),
    errorDetails: document.getElementById('pcd_error_details'),
};

// ===== Three.js 场景初始化 =====
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f172a);

const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 5000);
camera.up.set(0, 0, 1); // 雷达/工程坐标系 Z 轴朝上
camera.position.set(15, -15, 12);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
dom.canvasContainer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const cloudGroup = new THREE.Group();
const markerGroup = new THREE.Group();
scene.add(cloudGroup);
scene.add(markerGroup);

let axesHelper = new THREE.AxesHelper(5);
scene.add(axesHelper);

// 悬停高亮球
const hoverSphere = new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xfacc15, depthTest: false })
);
hoverSphere.visible = false;
hoverSphere.renderOrder = 999;
scene.add(hoverSphere);

const pcdLoader = new PCDLoader();
const raycaster = new THREE.Raycaster();
const pointerNDC = new THREE.Vector2();
const markerMap = new Map(); // pointId -> THREE.Group

function resizeRenderer() {
    const w = dom.canvasContainer.clientWidth;
    const h = dom.canvasContainer.clientHeight;
    if (w > 0 && h > 0) {
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    }
}
resizeRenderer();
if (typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(resizeRenderer).observe(dom.canvasContainer);
}
window.addEventListener('resize', resizeRenderer);

(function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
})();

// ===== 着色工具 =====
function jetColor(t, outColor) {
    const c = Math.max(0, Math.min(1, t));
    const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * c - 3)));
    const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * c - 2)));
    const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * c - 1)));
    outColor.setRGB(r, g, b);
}

function getCombinedBounds() {
    const box = new THREE.Box3();
    let hasAny = false;
    for (const f of state.frames) {
        if (f.visible && !f.bbox.isEmpty()) {
            box.union(f.bbox);
            hasAny = true;
        }
    }
    if (!hasAny) {
        for (const f of state.frames) {
            if (!f.bbox.isEmpty()) {
                box.union(f.bbox);
                hasAny = true;
            }
        }
    }
    return hasAny ? box : null;
}

function recolorAllFrames() {
    if (state.frames.length === 0) return;

    const bounds = getCombinedBounds();
    const minZ = bounds ? bounds.min.z : 0;
    const spanZ = bounds ? Math.max(bounds.max.z - bounds.min.z, 1e-3) : 1;

    let minI = Infinity;
    let maxI = -Infinity;
    let anyIntensity = false;
    if (state.colorMode === 'intensity') {
        for (const f of state.frames) {
            if (!f.rawIntensity) continue;
            anyIntensity = true;
            for (let i = 0; i < f.rawIntensity.length; i++) {
                const v = f.rawIntensity[i];
                if (v < minI) minI = v;
                if (v > maxI) maxI = v;
            }
        }
        if (!anyIntensity) {
            setHudText('当前点云不包含 intensity 字段，已自动按高度 Z 着色');
        }
    }
    const spanI = Math.max(maxI - minI, 1e-6);
    const tmpColor = new THREE.Color();

    for (const f of state.frames) {
        const count = f.pointCount;
        const colors = new Float32Array(count * 3);
        const pos = f.rawPositions;
        const useMode = (state.colorMode === 'intensity' && !f.hasIntensity) ? 'height' : state.colorMode;

        if (useMode === 'frame') {
            tmpColor.setHex(f.colorHex);
            for (let i = 0; i < count; i++) {
                colors[i * 3] = tmpColor.r;
                colors[i * 3 + 1] = tmpColor.g;
                colors[i * 3 + 2] = tmpColor.b;
            }
        } else if (useMode === 'intensity') {
            const inten = f.rawIntensity;
            for (let i = 0; i < count; i++) {
                jetColor((inten[i] - minI) / spanI, tmpColor);
                colors[i * 3] = tmpColor.r;
                colors[i * 3 + 1] = tmpColor.g;
                colors[i * 3 + 2] = tmpColor.b;
            }
        } else {
            for (let i = 0; i < count; i++) {
                jetColor((pos[i * 3 + 2] - minZ) / spanZ, tmpColor);
                colors[i * 3] = tmpColor.r;
                colors[i * 3 + 1] = tmpColor.g;
                colors[i * 3 + 2] = tmpColor.b;
            }
        }

        f.pointsObj.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        f.pointsObj.material.vertexColors = true;
        f.pointsObj.material.size = state.pointSize;
        f.pointsObj.material.needsUpdate = true;
    }
}

function updateSceneScaleAndCamera(resetCam = false) {
    const bounds = getCombinedBounds();
    if (!bounds) return;

    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    bounds.getSize(size);
    bounds.getCenter(center);
    const diag = Math.max(size.length(), 1.0);

    state.markerScale = Math.max(0.06, Math.min(0.6, diag * 0.008));
    hoverSphere.scale.setScalar(state.markerScale * 0.9);
    for (const grp of markerMap.values()) {
        grp.children[0].scale.setScalar(state.markerScale);
        grp.children[1].scale.set(state.markerScale * 4.5, state.markerScale * 2.25, 1);
        grp.children[1].position.set(0, 0, state.markerScale * 2.0);
    }

    scene.remove(axesHelper);
    axesHelper = new THREE.AxesHelper(Math.max(2, diag * 0.12));
    axesHelper.position.copy(center);
    scene.add(axesHelper);

    if (resetCam) {
        controls.target.copy(center);
        camera.position.set(
            center.x + diag * 0.65,
            center.y - diag * 0.65,
            center.z + diag * 0.45
        );
        camera.far = Math.max(5000, diag * 20);
        camera.updateProjectionMatrix();
        controls.update();
    }
}

function setHudText(text) {
    dom.hud.textContent = text;
}

// ===== 加载 PCD 文件 =====
function sanitizePointsObject(parsedPoints) {
    const geom = parsedPoints.geometry;
    const posAttr = geom.getAttribute('position');
    if (!posAttr) throw new Error('PCD 文件缺少位置属性 (x, y, z)');

    const colorAttr = geom.getAttribute('color');
    const intensityAttr = geom.getAttribute('intensity');

    const validPos = [];
    const validInten = [];
    const n = posAttr.count;
    for (let i = 0; i < n; i++) {
        const x = posAttr.getX(i);
        const y = posAttr.getY(i);
        const z = posAttr.getZ(i);
        if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
            validPos.push(x, y, z);
            if (intensityAttr) {
                const iv = intensityAttr.getX(i);
                validInten.push(Number.isFinite(iv) ? iv : 0);
            } else if (colorAttr) {
                // 部分 PCDLoader 将强度或灰阶存放在 color 中，仅作退路不强制当 intensity
            }
        }
    }

    if (validPos.length === 0) {
        throw new Error('PCD 文件中没有有效三维坐标点');
    }

    const cleanGeom = new THREE.BufferGeometry();
    const posArray = new Float32Array(validPos);
    cleanGeom.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    cleanGeom.computeBoundingBox();
    cleanGeom.computeBoundingSphere();

    const mat = new THREE.PointsMaterial({
        size: state.pointSize,
        sizeAttenuation: false,
        vertexColors: true,
    });
    const obj = new THREE.Points(cleanGeom, mat);
    return {
        pointsObj: obj,
        rawPositions: posArray,
        rawIntensity: intensityAttr ? new Float32Array(validInten) : null,
        pointCount: posArray.length / 3,
        bbox: cleanGeom.boundingBox.clone(),
    };
}

async function handleFilesUpload(fileList) {
    const files = Array.from(fileList || []);
    if (files.length === 0) return;

    const firstLoad = state.frames.length === 0;
    let loadedCount = 0;

    for (const file of files) {
        try {
            const buffer = await file.arrayBuffer();
            const parsed = pcdLoader.parse(buffer);
            const clean = sanitizePointsObject(parsed);
            const frameId = state.nextFrameId++;
            const colorHex = FRAME_PALETTE[(frameId - 1) % FRAME_PALETTE.length];

            clean.pointsObj.userData.frameName = file.name;
            clean.pointsObj.userData.frameId = frameId;
            cloudGroup.add(clean.pointsObj);

            state.frames.push({
                id: frameId,
                name: file.name,
                visible: true,
                colorHex,
                ...clean,
            });
            loadedCount++;
        } catch (err) {
            showError(`解析 ${file.name} 失败: ${err.message}`);
        }
    }

    if (loadedCount > 0) {
        renderFrameList();
        recolorAllFrames();
        updateSceneScaleAndCamera(firstLoad);
        const totalPts = state.frames.reduce((acc, f) => acc + f.pointCount, 0);
        setHudText(`已加载 ${state.frames.length} 帧点云，共 ${totalPts.toLocaleString()} 个点`);
    }
    dom.fileInput.value = '';
}

function renderFrameList() {
    if (state.frames.length === 0) {
        dom.frameList.innerHTML = '<span class="hint" style="margin:0;">尚未加载 PCD 点云文件</span>';
        return;
    }

    dom.frameList.innerHTML = '';
    for (const f of state.frames) {
        const chip = document.createElement('div');
        chip.className = 'frame-chip';
        const hexCss = '#' + f.colorHex.toString(16).padStart(6, '0');
        chip.innerHTML = `
            <label>
                <input type="checkbox" ${f.visible ? 'checked' : ''} data-frame-toggle="${f.id}">
                <span class="frame-color-dot" style="background:${hexCss};"></span>
                <span>${escapeHtml(f.name)} (${f.pointCount.toLocaleString()}点)</span>
            </label>
            <button type="button" title="移除此帧" data-frame-remove="${f.id}">×</button>
        `;
        dom.frameList.appendChild(chip);
    }
}

dom.frameList.addEventListener('change', (e) => {
    const id = Number(e.target.dataset.frameToggle);
    if (!id) return;
    const frame = state.frames.find(f => f.id === id);
    if (frame) {
        frame.visible = e.target.checked;
        frame.pointsObj.visible = frame.visible;
    }
});

dom.frameList.addEventListener('click', (e) => {
    const id = Number(e.target.dataset.frameRemove);
    if (!id) return;
    const idx = state.frames.findIndex(f => f.id === id);
    if (idx >= 0) {
        const [removed] = state.frames.splice(idx, 1);
        cloudGroup.remove(removed.pointsObj);
        removed.pointsObj.geometry.dispose();
        removed.pointsObj.material.dispose();
        renderFrameList();
        recolorAllFrames();
        updateSceneScaleAndCamera(false);
    }
});

dom.clearFramesBtn.addEventListener('click', () => {
    for (const f of state.frames) {
        cloudGroup.remove(f.pointsObj);
        f.pointsObj.geometry.dispose();
        f.pointsObj.material.dispose();
    }
    state.frames = [];
    hoverSphere.visible = false;
    renderFrameList();
    setHudText('已清空所有点云帧');
});

// ===== 3D 标记与拾取 =====
function getMarkerColor(pt) {
    if (!pt.enabled) return 0x64748b;
    if (pt.status === 'kept') return 0x22c55e;
    if (pt.status === 'removed') return 0xef4444;
    return 0x38bdf8;
}

function createLabelSprite(text, bgColor) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = bgColor;
    ctx.beginPath();
    ctx.roundRect(8, 8, 112, 48, 12);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 28px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 64, 34);

    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
    const sprite = new THREE.Sprite(mat);
    sprite.renderOrder = 1001;
    return sprite;
}

function syncPointMarker(pt) {
    let grp = markerMap.get(pt.id);
    const colorHex = getMarkerColor(pt);
    const cssColor = '#' + colorHex.toString(16).padStart(6, '0');

    if (!grp) {
        grp = new THREE.Group();
        const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(1, 16, 16),
            new THREE.MeshBasicMaterial({ color: colorHex, depthTest: false })
        );
        sphere.renderOrder = 1000;
        const sprite = createLabelSprite(`#${pt.id}`, cssColor);
        grp.add(sphere);
        grp.add(sprite);
        markerGroup.add(grp);
        markerMap.set(pt.id, grp);
    } else {
        grp.children[0].material.color.setHex(colorHex);
        // 更新标签颜色
        const oldSprite = grp.children[1];
        const newSprite = createLabelSprite(`#${pt.id}`, cssColor);
        oldSprite.material.map.dispose();
        oldSprite.material.dispose();
        grp.remove(oldSprite);
        grp.add(newSprite);
    }

    grp.position.set(pt.radar[0], pt.radar[1], pt.radar[2]);
    grp.children[0].scale.setScalar(state.markerScale);
    grp.children[1].scale.set(state.markerScale * 4.5, state.markerScale * 2.25, 1);
    grp.children[1].position.set(0, 0, state.markerScale * 2.0);
}

function removePointMarker(pointId) {
    const grp = markerMap.get(pointId);
    if (!grp) return;
    markerGroup.remove(grp);
    grp.children[0].geometry.dispose();
    grp.children[0].material.dispose();
    grp.children[1].material.map.dispose();
    grp.children[1].material.dispose();
    markerMap.delete(pointId);
}

function syncAllMarkers() {
    const validIds = new Set(state.calibPoints.map(p => p.id));
    for (const pid of Array.from(markerMap.keys())) {
        if (!validIds.has(pid)) removePointMarker(pid);
    }
    for (const pt of state.calibPoints) {
        syncPointMarker(pt);
    }
}

function isPickingActive() {
    return state.pickMode || state.shiftDown;
}

function updatePickUI() {
    dom.modeViewBtn.classList.toggle('active', !state.pickMode);
    dom.modePickBtn.classList.toggle('active', state.pickMode);
    dom.viewerWrap.classList.toggle('pick-cursor', isPickingActive());
}

function raycastVisibleClouds(clientX, clientY) {
    const visibleObjs = state.frames.filter(f => f.visible).map(f => f.pointsObj);
    if (visibleObjs.length === 0) return null;

    const rect = renderer.domElement.getBoundingClientRect();
    pointerNDC.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointerNDC.y = -((clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(pointerNDC, camera);
    raycaster.params.Points.threshold = state.pickThreshold;

    const hits = raycaster.intersectObjects(visibleObjs, false);
    if (hits.length === 0) return null;

    const hit = hits[0];
    const posAttr = hit.object.geometry.getAttribute('position');
    const idx = hit.index;
    return {
        x: posAttr.getX(idx),
        y: posAttr.getY(idx),
        z: posAttr.getZ(idx),
        frameName: hit.object.userData.frameName || '未知帧',
    };
}

let pointerDownPos = null;

renderer.domElement.addEventListener('pointermove', (e) => {
    if (state.frames.length === 0) return;
    const hit = raycastVisibleClouds(e.clientX, e.clientY);
    if (hit) {
        hoverSphere.position.set(hit.x, hit.y, hit.z);
        hoverSphere.visible = true;
        setHudText(
            `[${hit.frameName}] X=${hit.x.toFixed(4)}, Y=${hit.y.toFixed(4)}, Z=${hit.z.toFixed(4)}` +
            (isPickingActive() ? '  (点击拾取为标定点)' : '')
        );
    } else {
        hoverSphere.visible = false;
    }
});

renderer.domElement.addEventListener('pointerdown', (e) => {
    if (e.button === 0) {
        pointerDownPos = { x: e.clientX, y: e.clientY };
    }
});

renderer.domElement.addEventListener('pointerup', (e) => {
    if (e.button !== 0 || !pointerDownPos) return;
    const dist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
    pointerDownPos = null;
    if (dist > 5 || !isPickingActive()) return;

    const hit = raycastVisibleClouds(e.clientX, e.clientY);
    if (!hit) {
        setHudText('未命中点云，请靠近点云或适当调大“拾取半径”');
        return;
    }
    addCalibPoint([hit.x, hit.y, hit.z], hit.frameName);
});

window.addEventListener('keydown', (e) => {
    if (e.key === 'Shift') {
        state.shiftDown = true;
        updatePickUI();
    }
});

window.addEventListener('keyup', (e) => {
    if (e.key === 'Shift') {
        state.shiftDown = false;
        updatePickUI();
    }
});

// ===== 标定点管理与表格渲染 =====
function resetPointStatuses() {
    for (const pt of state.calibPoints) {
        pt.status = 'pending';
        pt.error = null;
        pt.removedRound = null;
    }
}

function addCalibPoint(radar, frameName = '手动输入', blh = ['', '', '']) {
    const id = state.nextPointId++;
    state.calibPoints.push({
        id,
        frameName,
        radar: radar.map(v => Number(Number(v).toFixed(6))),
        blh: [...blh],
        enabled: true,
        status: 'pending',
        error: null,
        removedRound: null,
    });
    renderPointsTable();
    syncAllMarkers();
}

function renderPointsTable() {
    const enabledCount = state.calibPoints.filter(p => p.enabled).length;
    const minPts = Math.max(3, Number(document.getElementById('pcd_min_points')?.value) || 3);
    dom.pointsCountHint.textContent =
        `共 ${state.calibPoints.length} 组点（已启用 ${enabledCount} 组，至少需要 ${minPts} 组）`;

    if (state.calibPoints.length === 0) {
        dom.pointsTbody.innerHTML =
            '<tr><td colspan="11" class="hint" style="padding:16px;">暂无标定点，请在上方点云中拾取或点击“手动添加点”。</td></tr>';
        return;
    }

    dom.pointsTbody.innerHTML = '';
    for (const pt of state.calibPoints) {
        const tr = document.createElement('tr');
        if (!pt.enabled) tr.classList.add('row-disabled');
        else if (pt.status === 'removed') tr.classList.add('row-removed');

        let statusHtml = '<span class="badge badge-pending">待计算</span>';
        if (pt.status === 'kept') {
            statusHtml = `<span class="badge badge-kept">保留 (${pt.error.toFixed(4)}m)</span>`;
        } else if (pt.status === 'removed') {
            statusHtml = `<span class="badge badge-removed">第${pt.removedRound}轮剔除 (${pt.error.toFixed(4)}m)</span>`;
        }

        tr.innerHTML = `
            <td><input type="checkbox" ${pt.enabled ? 'checked' : ''} data-pt-enable="${pt.id}"></td>
            <td><strong>#${pt.id}</strong></td>
            <td title="${escapeHtml(pt.frameName)}">${escapeHtml(truncateName(pt.frameName, 16))}</td>
            <td><input type="number" step="0.0001" value="${pt.radar[0]}" data-pt-field="radar0" data-pt-id="${pt.id}"></td>
            <td><input type="number" step="0.0001" value="${pt.radar[1]}" data-pt-field="radar1" data-pt-id="${pt.id}"></td>
            <td><input type="number" step="0.0001" value="${pt.radar[2]}" data-pt-field="radar2" data-pt-id="${pt.id}"></td>
            <td><input type="number" class="input-bl" step="0.000000001" placeholder="纬度 B" value="${pt.blh[0]}" data-pt-field="blh0" data-pt-id="${pt.id}"></td>
            <td><input type="number" class="input-bl" step="0.000000001" placeholder="经度 L" value="${pt.blh[1]}" data-pt-field="blh1" data-pt-id="${pt.id}"></td>
            <td><input type="number" step="0.0001" placeholder="海拔 H" value="${pt.blh[2]}" data-pt-field="blh2" data-pt-id="${pt.id}"></td>
            <td>${statusHtml}</td>
            <td>
                <button type="button" class="btn-sm" data-pt-focus="${pt.id}">定位</button>
                <button type="button" class="btn-sm btn-danger-sm" data-pt-del="${pt.id}">删除</button>
            </td>
        `;
        dom.pointsTbody.appendChild(tr);
    }
}

dom.pointsTbody.addEventListener('input', (e) => {
    const id = Number(e.target.dataset.ptId);
    const field = e.target.dataset.ptField;
    if (!id || !field) return;
    const pt = state.calibPoints.find(p => p.id === id);
    if (!pt) return;

    const val = e.target.value;
    if (field.startsWith('radar')) {
        const axis = Number(field.slice(5));
        const num = Number(val);
        if (val !== '' && Number.isFinite(num)) {
            pt.radar[axis] = num;
            syncPointMarker(pt);
        }
    } else if (field.startsWith('blh')) {
        const axis = Number(field.slice(3));
        pt.blh[axis] = val;
    }
});

dom.pointsTbody.addEventListener('change', (e) => {
    const id = Number(e.target.dataset.ptEnable);
    if (!id) return;
    const pt = state.calibPoints.find(p => p.id === id);
    if (pt) {
        pt.enabled = e.target.checked;
        resetPointStatuses();
        renderPointsTable();
        syncAllMarkers();
    }
});

dom.pointsTbody.addEventListener('click', (e) => {
    const focusId = Number(e.target.dataset.ptFocus);
    if (focusId) {
        const pt = state.calibPoints.find(p => p.id === focusId);
        if (pt) {
            const target = new THREE.Vector3(pt.radar[0], pt.radar[1], pt.radar[2]);
            const offset = camera.position.clone().sub(controls.target);
            if (offset.length() > 15) offset.setLength(10);
            controls.target.copy(target);
            camera.position.copy(target).add(offset);
            controls.update();
        }
        return;
    }

    const delId = Number(e.target.dataset.ptDel);
    if (delId) {
        state.calibPoints = state.calibPoints.filter(p => p.id !== delId);
        renderPointsTable();
        syncAllMarkers();
    }
});

// ===== 工具栏与控件事件 =====
dom.fileInput.addEventListener('change', (e) => handleFilesUpload(e.target.files));
dom.modeViewBtn.addEventListener('click', () => { state.pickMode = false; updatePickUI(); });
dom.modePickBtn.addEventListener('click', () => { state.pickMode = true; updatePickUI(); });

dom.colorMode.addEventListener('change', (e) => {
    state.colorMode = e.target.value;
    recolorAllFrames();
});

dom.pointSize.addEventListener('input', (e) => {
    state.pointSize = Number(e.target.value);
    for (const f of state.frames) {
        f.pointsObj.material.size = state.pointSize;
        f.pointsObj.material.needsUpdate = true;
    }
});

dom.pickThreshold.addEventListener('input', (e) => {
    state.pickThreshold = Number(e.target.value);
    dom.pickThresholdVal.textContent = `${state.pickThreshold.toFixed(2)}m`;
});

dom.resetViewBtn.addEventListener('click', () => updateSceneScaleAndCamera(true));

dom.topViewBtn.addEventListener('click', () => {
    const bounds = getCombinedBounds();
    if (!bounds) return;
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    bounds.getCenter(center);
    bounds.getSize(size);
    const diag = Math.max(size.length(), 2.0);
    controls.target.copy(center);
    camera.position.set(center.x, center.y - 0.01, center.z + diag * 1.1);
    controls.update();
});

dom.addManualBtn.addEventListener('click', () => {
    addCalibPoint([0, 0, 0], '手动输入');
});

dom.clearPointsBtn.addEventListener('click', () => {
    state.calibPoints = [];
    renderPointsTable();
    syncAllMarkers();
    dom.resultSection.style.display = 'none';
});

dom.exportJsonBtn.addEventListener('click', () => {
    const data = state.calibPoints.map(p => ({
        id: p.id,
        frameName: p.frameName,
        radar: p.radar,
        blh: p.blh,
        enabled: p.enabled,
    }));
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'calibration_points.json';
    a.click();
    URL.revokeObjectURL(url);
});

dom.importJsonBtn.addEventListener('click', () => dom.importJsonInput.click());
dom.importJsonInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
        const text = await file.text();
        const items = JSON.parse(text);
        if (!Array.isArray(items)) throw new Error('JSON 顶层必须为数组');

        state.calibPoints = [];
        let maxId = 0;
        for (const item of items) {
            const id = Number(item.id) || (maxId + 1);
            if (id > maxId) maxId = id;
            const radar = Array.isArray(item.radar) && item.radar.length === 3
                ? item.radar.map(Number) : [0, 0, 0];
            const blh = Array.isArray(item.blh) && item.blh.length === 3
                ? item.blh : ['', '', ''];
            state.calibPoints.push({
                id,
                frameName: item.frameName || '导入点',
                radar,
                blh,
                enabled: item.enabled !== false,
                status: 'pending',
                error: null,
                removedRound: null,
            });
        }
        state.nextPointId = maxId + 1;
        renderPointsTable();
        syncAllMarkers();
    } catch (err) {
        showError(`导入标定点失败: ${err.message}`);
    }
    dom.importJsonInput.value = '';
});

// 坐标系联动
function updatePcdCoordVisibility() {
    const cs = dom.coordSystem.value;
    dom.armParams.style.display = (cs === 'arm' || cs === 'chute') ? 'block' : 'none';
    dom.connectParams.style.display = (cs === 'chute') ? 'block' : 'none';
}
dom.coordSystem.addEventListener('change', updatePcdCoordVisibility);
updatePcdCoordVisibility();

// ===== 提交迭代标定 =====
function readRequiredNum(id, label) {
    const raw = document.getElementById(id).value;
    if (raw === '' || raw === null) {
        throw new Error(`请填写 ${label}`);
    }
    const num = Number(raw);
    if (!Number.isFinite(num)) {
        throw new Error(`${label} 必须为有效数字`);
    }
    return num;
}

dom.calcBtn.addEventListener('click', async () => {
    hideError();

    try {
        const minPoints = Math.max(3, Math.trunc(readRequiredNum('pcd_min_points', '最少保留点数')));
        const errorThreshold = readRequiredNum('pcd_error_threshold', '允许最大误差阈值');
        if (errorThreshold <= 0) throw new Error('允许最大误差阈值必须大于 0');

        const activePoints = state.calibPoints.filter(p => p.enabled);
        if (activePoints.length < minPoints) {
            throw new Error(`启用的标定点数量 (${activePoints.length}) 少于最少保留点数 (${minPoints})`);
        }

        const payloadPoints = activePoints.map(pt => {
            const rx = Number(pt.radar[0]);
            const ry = Number(pt.radar[1]);
            const rz = Number(pt.radar[2]);
            if (![rx, ry, rz].every(Number.isFinite)) {
                throw new Error(`标定点 #${pt.id} 的雷达坐标无效`);
            }
            if (pt.blh.some(v => v === '' || v === null || v === undefined)) {
                throw new Error(`请填写标定点 #${pt.id} 的完整纬度 B、经度 L 和海拔 H`);
            }
            const b = Number(pt.blh[0]);
            const l = Number(pt.blh[1]);
            const h = Number(pt.blh[2]);
            if (![b, l, h].every(Number.isFinite)) {
                throw new Error(`标定点 #${pt.id} 的经纬度或海拔数值无效`);
            }
            return { id: pt.id, radar: [rx, ry, rz], blh: [b, l, h] };
        });

        const transformMatrix = document.getElementById('pcd_transform_matrix').value.trim();
        if (!transformMatrix) {
            throw new Error('请输入 WGS84 到泊位坐标系的 4×4 转换矩阵');
        }

        const coordSystem = dom.coordSystem.value;
        const params = {
            slewing_angle: readRequiredNum('pcd_slewing_angle', '回转角'),
            luffing_angle: readRequiredNum('pcd_luffing_angle', '俯仰角'),
            rot_center: [
                readRequiredNum('pcd_rot_center_x', '旋转中心 X'),
                readRequiredNum('pcd_rot_center_y', '旋转中心 Y'),
                readRequiredNum('pcd_rot_center_z', '旋转中心 Z'),
            ],
            arm_length: coordSystem === 'rot_center' ? 25.3 : readRequiredNum('pcd_arm_length', '臂架长度'),
            stretch_dist: coordSystem === 'rot_center' ? 0.0 : readRequiredNum('pcd_stretch_dist', '伸缩距离'),
            connect_dist: coordSystem === 'chute' ? readRequiredNum('pcd_connect_dist', '连接点距离') : 0.0,
        };

        const body = {
            points: payloadPoints,
            undo: readRequiredNum('pcd_undo', '高程差 undo'),
            transform_matrix: transformMatrix,
            coord_system: coordSystem,
            params,
            error_threshold: errorThreshold,
            min_points: minPoints,
        };

        dom.calcBtn.disabled = true;
        dom.calcBtn.textContent = '计算中...';

        const resp = await fetch('/calculate_points', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await resp.json();

        if (!resp.ok || data.error) {
            throw new Error(data.error || `请求失败 (${resp.status})`);
        }

        applyCalibResult(data);
    } catch (err) {
        showError(err.message);
    } finally {
        dom.calcBtn.disabled = false;
        dom.calcBtn.textContent = '开始迭代标定';
    }
});

function applyCalibResult(data) {
    // 1. 回写点状态到列表和 3D 标记
    resetPointStatuses();
    const resMap = new Map(data.point_results.map(r => [r.id, r]));
    for (const pt of state.calibPoints) {
        const res = resMap.get(pt.id);
        if (res) {
            pt.status = res.status;
            pt.error = res.error;
            pt.removedRound = res.removed_round;
        }
    }
    renderPointsTable();
    syncAllMarkers();

    // 2. 状态横幅
    dom.resultSection.style.display = 'block';
    dom.calibBanner.className = 'calib-banner ' + (data.converged ? 'ok' : 'warn');
    dom.calibBanner.textContent = data.message;

    // 3. 标定矩阵表格 + 高精度文本
    const matrix = data.matrix;
    let matrixHtml = '<table class="matrix-table">';
    for (let i = 0; i < 4; i++) {
        matrixHtml += '<tr>';
        for (let j = 0; j < 4; j++) {
            matrixHtml += `<td>${matrix[i][j].toFixed(6)}</td>`;
        }
        matrixHtml += '</tr>';
    }
    matrixHtml += '</table>';
    dom.matrixResult.innerHTML = matrixHtml;

    const pyRows = matrix.map(row => ' [' + row.map(v => String(v)).join(', ') + ']');
    dom.matrixNumpy.textContent = '[' + pyRows.join(',\n').trimStart() + ']';

    // 4. 迭代过程表
    dom.iterTbody.innerHTML = '';
    for (const step of data.iterations) {
        const tr = document.createElement('tr');
        const actionText = step.removed_id !== null
            ? `剔除 点 #${step.removed_id} (误差 ${step.removed_error.toFixed(4)} m)`
            : (data.converged ? '全部点满足阈值，收敛结束' : '达到最少点数，停止迭代');
        tr.innerHTML = `
            <td>第 ${step.round} 轮</td>
            <td>${step.active_count}</td>
            <td>${step.mean_error.toFixed(6)}</td>
            <td>${step.max_error.toFixed(6)}</td>
            <td>${actionText}</td>
        `;
        dom.iterTbody.appendChild(tr);
    }

    // 5. 误差汇总与逐点明细
    dom.errorSummary.textContent =
        `保留点数: ${data.kept_count} / ${data.total_count} | 保留点平均误差: ${data.mean_error.toFixed(4)} m | 保留点最大误差: ${data.max_error.toFixed(4)} m`;

    let errHtml = '<table class="error-table"><tr><th>标定点</th><th>状态</th><th>误差 (m)</th></tr>';
    for (const pr of data.point_results) {
        const stateLabel = pr.status === 'kept'
            ? '<span class="badge badge-kept">保留</span>'
            : `<span class="badge badge-removed">第 ${pr.removed_round} 轮剔除</span>`;
        errHtml += `<tr><td>点 #${pr.id}</td><td>${stateLabel}</td><td>${pr.error.toFixed(6)}</td></tr>`;
    }
    errHtml += '</table>';
    dom.errorDetails.innerHTML = errHtml;
}

dom.copyMatrixBtn.addEventListener('click', async () => {
    try {
        await navigator.clipboard.writeText(dom.matrixNumpy.textContent);
        dom.copyMatrixBtn.textContent = '已复制';
        setTimeout(() => { dom.copyMatrixBtn.textContent = '复制矩阵'; }, 1500);
    } catch {
        showError('浏览器剪贴板复制失败，请手动选中文本复制');
    }
});

function showError(msg) {
    dom.errorBox.textContent = msg;
    dom.errorBox.style.display = 'block';
}

function hideError() {
    dom.errorBox.style.display = 'none';
}

function truncateName(str, maxLen) {
    return str.length > maxLen ? str.slice(0, maxLen - 1) + '…' : str;
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
