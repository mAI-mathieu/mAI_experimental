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
    const count = Number(node.widgets.find((w) => w.name === "reference_count")?.value ?? 2);
    const json = String(node.widgets.find((w) => w.name === "reference_json")?.value ?? "").trim();
    for (const row of node.maiImageRows ?? []) {
        for (const widget of row.widgets) setVisible(widget, !json && row.index <= count);
    }
    // IMAGE sockets stay in place so links and saved socket indexes remain stable.
    if (resize) node.setSize([Math.max(node.size[0], 380), node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
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
                if (!file || !time) continue;
                const button = this.addWidget("button", `Upload image ${index}`, null, () => chooseImage(this, file), { serialize: false });
                const preview = addPreview(this, index, file);
                this.maiImageRows.push({ index, widgets: [file, time, button, preview] });
            }
            // Put each uploader and preview directly after its image/time fields.
            const rowWidgets = new Set(this.maiImageRows.flatMap((row) => row.widgets));
            const ordered = this.widgets.filter((widget) => !rowWidgets.has(widget));
            ordered.push(...this.maiImageRows.flatMap((row) => row.widgets));
            this.widgets.splice(0, this.widgets.length, ...ordered);
            for (const name of ["reference_count", "reference_json"]) {
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
            updateRows(this, false);
            return result;
        };
    },
});
