import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "MAIH3Multiframe";
const MAX_IMAGES = 9;
const hiddenWidgets = new WeakMap();

function setVisible(widget, visible) {
    if (visible && hiddenWidgets.has(widget)) {
        const saved = hiddenWidgets.get(widget);
        widget.type = saved.type;
        widget.computeSize = saved.computeSize;
        hiddenWidgets.delete(widget);
    } else if (!visible && !hiddenWidgets.has(widget)) {
        hiddenWidgets.set(widget, { type: widget.type, computeSize: widget.computeSize });
        widget.type = "mai_hidden";
        widget.computeSize = () => [0, -4];
    }
    widget.hidden = !visible;
}

function updateRows(node, resize = true) {
    if (!node.maiImageRows || node.maiUpdatingRows) return;
    node.maiUpdatingRows = true;
    try {
        const json = String(node.widgets.find((w) => w.name === "reference_json")?.value ?? "").trim();
        const connected = new Set(node.inputs
            .filter((slot) => slot.link != null && /^image_\d+$/.test(slot.name))
            .map((slot) => Number(slot.name.slice(6))));
        const occupied = node.maiImageRows.filter((row) => {
            const path = row.widgets[0].value;
            return connected.has(row.index) || (path && path !== "(none)");
        }).map((row) => row.index);
        const last = Math.max(0, ...occupied);
        // Prefer a spare after the existing rows. Reuse earlier holes at the limit.
        const spare = node.maiImageRows.find((row) => row.index > last)
            ?? node.maiImageRows.find((row) => !occupied.includes(row.index));
        const visible = new Set(occupied);
        if (spare && !json) visible.add(spare.index);
        const referenceDefault = node.widgets.find((widget) => widget.name === "use_as_references")?.value ?? true;
        for (const row of node.maiImageRows) {
            const mode = row.reference?.value ?? "default";
            const isReference = mode === "yes" || (mode === "default" && referenceDefault);
            for (const widget of row.widgets) {
                setVisible(widget, !json && visible.has(row.index) && (widget !== row.time || !isReference));
            }
        }
        // Remove only unused IMAGE sockets. LiteGraph updates remaining link indexes.
        // Connected sockets survive JSON mode so returning to rows restores the links.
        for (let slot = node.inputs.length - 1; slot >= 0; slot--) {
            const input = node.inputs[slot];
            if (/^image_\d+$/.test(input.name)) {
                const index = Number(input.name.slice(6));
                if (!connected.has(index) && (json || !visible.has(index))) node.removeInput(slot);
            }
        }
        if (!json) {
            for (const row of node.maiImageRows) {
                if (visible.has(row.index) && !node.inputs.some((slot) => slot.name === `image_${row.index}`)) {
                    node.addInput(`image_${row.index}`, "IMAGE");
                }
            }
        }
        // Keep the released backend field for old workflows and API clients.
        const count = node.widgets.find((widget) => widget.name === "reference_count");
        if (count) {
            count.value = Math.max(1, last);
            setVisible(count, false);
        }
        if (resize) node.setSize([Math.max(node.size[0], 380), node.computeSize()[1]]);
        node.setDirtyCanvas(true, true);
    } finally {
        node.maiUpdatingRows = false;
    }
}

function scheduleRows(node) {
    if (node.maiUpdatingRows || node.maiRowsScheduled) return;
    node.maiRowsScheduled = true;
    // Connection callbacks can fire partway through LiteGraph's link updates.
    queueMicrotask(() => {
        node.maiRowsScheduled = false;
        updateRows(node);
    });
}

function removeImage(node, row) {
    const slot = node.inputs.findIndex((input) => input.name === `image_${row.index}`);
    if (slot >= 0 && node.inputs[slot].link != null) node.disconnectInput(slot);
    const file = row.widgets[0];
    file.value = "(none)";
    file.callback?.(file.value);
    // Other rows retain their file selections, timestamps, names, and links.
    updateRows(node);
}

async function uploadImage(node, fileWidget, file) {
    const body = new FormData();
    body.append("image", file);
    body.append("type", "input");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) throw new Error(`Image upload failed (${response.status}): ${await response.text()}`);
    const result = await response.json();
    const path = result.subfolder ? `${result.subfolder}/${result.name}` : result.name;
    // Make the uploaded file available in every row, allowing it to be reused.
    for (const widget of node.widgets) {
        if (/^image_file_\d+$/.test(widget.name) && Array.isArray(widget.options.values) && !widget.options.values.includes(path)) {
            widget.options.values.push(path);
        }
    }
    fileWidget.value = path;
    fileWidget.callback?.(path);
    node.setDirtyCanvas(true, true);
}

function chooseImage(node, widget) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/png,image/jpeg,image/webp,image/bmp,image/tiff";
    input.onchange = async () => {
        try {
            if (input.files?.[0]) await uploadImage(node, widget, input.files[0]);
        } catch (error) {
            window.alert(error.message);
        } finally {
            input.remove();
        }
    };
    input.oncancel = () => input.remove();
    input.style.display = "none";
    document.body.appendChild(input);
    input.click();
}

function addPreview(node, index, fileWidget) {
    let image;
    let currentPath;
    const preview = {
        name: `preview_${index}`, type: "mai_image_preview", value: "",
        options: { serialize: false },
        computeSize: () => [360, 100],
        draw(ctx, owner, width, y) {
            const path = fileWidget.value;
            if (currentPath !== path) {
                currentPath = path;
                image = undefined;
                if (path && path !== "(none)") {
                    const normalized = path.replaceAll("\\", "/");
                    const parts = normalized.split("/");
                    const filename = parts.pop();
                    const next = new Image();
                    next.onload = () => {
                        if (currentPath === path) { image = next; owner.setDirtyCanvas(true); }
                    };
                    next.src = api.apiURL(`/view?${new URLSearchParams({ filename, subfolder: parts.join("/"), type: "input" })}`);
                }
            }
            ctx.save();
            ctx.fillStyle = "#202020";
            ctx.fillRect(10, y + 4, width - 20, 92);
            const connected = owner.inputs?.find((slot) => slot.name === `image_${index}`)?.link != null;
            if (image && !connected) {
                const scale = Math.min((width - 24) / image.width, 88 / image.height);
                ctx.drawImage(image, (width - image.width * scale) / 2, y + 6 + (88 - image.height * scale) / 2, image.width * scale, image.height * scale);
            } else {
                ctx.fillStyle = "#bbb";
                ctx.textAlign = "center";
                ctx.fillText(connected ? `Using connected image_${index}` : `Image ${index}: select a file or connect an IMAGE`, width / 2, y + 53);
            }
            ctx.restore();
        },
    };
    node.addCustomWidget(preview);
    return preview;
}

app.registerExtension({
    name: "mAI.H3Multiframe",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            this.maiImageRows = [];
            for (let index = 1; index <= MAX_IMAGES; index++) {
                const file = this.widgets.find((w) => w.name === `image_file_${index}`);
                const time = this.widgets.find((w) => w.name === `time_${index}`);
                const reference = this.widgets.find((w) => w.name === `use_as_reference_${index}`);
                if (!file || !time) continue;
                const button = this.addWidget("button", `Upload image ${index}`, null, () => chooseImage(this, file), { serialize: false });
                const preview = addPreview(this, index, file);
                const widgets = [file];
                if (reference) widgets.push(reference);
                widgets.push(time, button, preview);
                const row = { index, time, reference, widgets };
                const remove = this.addWidget("button", `Remove image ${index}`, null, () => removeImage(this, row), { serialize: false });
                row.widgets.push(remove);
                this.maiImageRows.push(row);
                const callback = file.callback;
                file.callback = (...args) => { callback?.apply(file, args); updateRows(this); };
                if (reference) {
                    const referenceCallback = reference.callback;
                    reference.callback = (...args) => { referenceCallback?.apply(reference, args); updateRows(this); };
                }
            }
            // Put each uploader and preview directly after its image/time fields.
            const rowWidgets = new Set(this.maiImageRows.flatMap((row) => row.widgets));
            const ordered = this.widgets.filter((widget) => !rowWidgets.has(widget));
            ordered.push(...this.maiImageRows.flatMap((row) => row.widgets));
            this.widgets.splice(0, this.widgets.length, ...ordered);
            const referenceDefault = this.widgets.find((w) => w.name === "use_as_references");
            if (referenceDefault) referenceDefault.label = "default use as reference";
            for (const name of ["reference_count", "reference_json", "use_as_references"]) {
                const widget = this.widgets.find((w) => w.name === name);
                const callback = widget.callback;
                widget.callback = (...args) => { callback?.apply(widget, args); updateRows(this); };
            }
            updateRows(this);
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            scheduleRows(this);
            return result;
        };
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            scheduleRows(this);
            return result;
        };
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function (data) {
            // Inserting per-picture roles must not shift old file/time values.
            const persistent = this.widgets.filter((w) => w.serialize !== false && w.options?.serialize !== false);
            const legacy = persistent.filter((w) => !/^use_as_reference_\d+$/.test(w.name));
            const saved = data.widgets_values;
            let named = data.widgets_values_named;
            if (!named && Array.isArray(saved) && saved.length === legacy.length) {
                named = Object.fromEntries(legacy.map((widget, index) => [widget.name, saved[index]]));
            }
            if (named) {
                data = { ...data, widgets_values_named: { ...named }, widgets_values:
                    persistent.map((widget) => Object.hasOwn(named, widget.name) ? named[widget.name] : widget.value) };
            }
            if (Array.isArray(data.inputs)) {
                // ComfyUI merges default inputs before restoring saved ones. Match
                // the saved order first, or a removed default spare could shift
                // the target indexes of restored image links.
                const existing = new Map(this.inputs.map((input) => [input.name, input]));
                this.inputs.splice(0, this.inputs.length, ...data.inputs.map((input) =>
                    existing.get(input.name) ?? { ...input, link: null }));
            }
            return configure?.apply(this, arguments);
        };
    },
});
