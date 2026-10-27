import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/js/h3_multiframe.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");

async function createHarness() {
    let extension;
    let picker;
    let nextLink = 1;
    const links = new Map();
    const pending = [];
    const api = {
        apiURL: (url) => url,
        async fetchApi(url) {
            assert.equal(url, "/upload/image");
            return { ok: true, async json() { return { name: "uploaded.png", subfolder: "refs" }; } };
        },
    };
    vm.runInNewContext(source, {
        app: { registerExtension(value) { extension = value; } },
        api, URLSearchParams,
        queueMicrotask(callback) { pending.push(callback); },
        Image: class {},
        FormData: class { append() {} },
        window: { alert(message) { assert.fail(message); } },
        document: {
            body: { appendChild() {} },
            createElement() {
                picker = { style: {}, files: ["image file"], click() {}, remove() {} };
                return picker;
            },
        },
    });

    class Node {
        constructor() {
            this.size = [400, 600];
            this.inputs = [{ name: "clip", type: "CLIP", link: null }, { name: "vae", type: "VAE", link: null }];
            for (let index = 1; index <= 9; index++) this.inputs.push({ name: `image_${index}`, type: "IMAGE", link: null });
            this.widgets = [
                { name: "reference_count", type: "number", value: 2 },
                { name: "reference_json", type: "text", value: "" },
            ];
            for (let index = 1; index <= 9; index++) {
                this.widgets.push(
                    { name: `image_file_${index}`, type: "combo", value: "(none)", options: { values: ["(none)"] } },
                    { name: `time_${index}`, type: "number", value: (index - 1) * 1.5 },
                );
            }
            this.onNodeCreated();
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        }
        addCustomWidget(widget) { this.widgets.push(widget); return widget; }
        addInput(name, type) {
            this.inputs.push({ name, type, link: null });
            this.onConnectionsChange();
        }
        removeInput(index) {
            assert.equal(this.inputs[index].link, null, "Do not remove a connected socket");
            this.inputs.splice(index, 1);
            for (const link of links.values()) if (link.target_slot > index) link.target_slot--;
            this.onConnectionsChange();
        }
        disconnectInput(index) {
            links.delete(this.inputs[index].link);
            this.inputs[index].link = null;
            this.onConnectionsChange();
        }
        configure(data) {
            // Mirror ComfyUI's default-first input merge during graph restoration.
            const names = new Set(this.inputs.map((input) => input.name));
            this.inputs = this.inputs.map((input) => ({ ...input, ...data.inputs.find((saved) => saved.name === input.name) }))
                .concat(data.inputs.filter((input) => !names.has(input.name)));
            Object.assign(this.widgets.find((widget) => widget.name === "reference_count"), { value: data.count ?? 2 });
            for (const [name, value] of Object.entries(data.values ?? {})) {
                this.widgets.find((widget) => widget.name === name).value = value;
            }
            this.onConfigure();
        }
        setSize(size) { this.size = size; }
        computeSize() { return [400, 200 + this.widgets.filter((widget) => !widget.hidden).length * 20]; }
        setDirtyCanvas() {}
    }

    await extension.beforeRegisterNodeDef(Node, { name: "MAIH3Multiframe" });
    const flush = () => { while (pending.length) pending.shift()(); };
    const node = new Node();
    flush();
    const widget = (name, target = node) => target.widgets.find((item) => item.name === name);
    return {
        node, Node, links, flush, widget,
        imageInputs(target = node) { return target.inputs.filter((input) => /^image_\d+$/.test(input.name)).map((input) => input.name); },
        visibleRows(target = node) { return Array.from(target.maiImageRows).filter((row) => !row.widgets[0].hidden).map((row) => row.index); },
        set(name, value) { widget(name).value = value; widget(name).callback?.(value); flush(); },
        connect(name) {
            const target_slot = node.inputs.findIndex((input) => input.name === name);
            assert(target_slot >= 0);
            const id = nextLink++;
            links.set(id, { id, target_slot });
            // Fire before assignment to verify the deferred connection refresh.
            node.onConnectionsChange();
            node.inputs[target_slot].link = id;
            flush();
            return id;
        },
        disconnect(name) { node.disconnectInput(node.inputs.findIndex((input) => input.name === name)); flush(); },
        remove(index) { widget(`Remove image ${index}`).callback(); flush(); },
        async upload(index) { widget(`Upload image ${index}`).callback(); await picker.onchange(); flush(); },
    };
}

test("one spare initially; file selection and clearing grow and shrink entries", async () => {
    const h = await createHarness();
    assert.deepEqual(h.visibleRows(), [1]);
    assert.deepEqual(h.imageInputs(), ["image_1"]);
    assert.equal(h.widget("reference_count").hidden, true);
    h.set("image_file_1", "a.png");
    assert.deepEqual(h.visibleRows(), [1, 2]);
    h.set("image_file_2", "b.png");
    assert.deepEqual(h.imageInputs(), ["image_1", "image_2", "image_3"]);
    h.set("image_file_2", "(none)");
    assert.deepEqual(h.visibleRows(), [1, 2]);
    h.set("image_file_1", "(none)");
    assert.deepEqual(h.imageInputs(), ["image_1"]);
});

test("uploads grow entries and expose the uploaded file in every selector", async () => {
    const h = await createHarness();
    await h.upload(1);
    assert.deepEqual(h.imageInputs(), ["image_1", "image_2"]);
    assert.equal(h.widget("image_file_1").value, "refs/uploaded.png");
    assert(h.node.maiImageRows.every((row) => row.widgets[0].options.values.includes("refs/uploaded.png")));
});

test("connections grow entries; middle removal preserves remaining links and times", async () => {
    const h = await createHarness();
    h.connect("image_1");
    h.connect("image_2");
    const third = h.connect("image_3");
    h.set("time_3", 4.5);
    h.remove(2);
    assert.deepEqual(h.visibleRows(), [1, 3, 4]);
    assert.deepEqual(h.imageInputs(), ["image_1", "image_3", "image_4"]);
    assert.equal(h.node.inputs[h.links.get(third).target_slot].name, "image_3");
    assert.equal(h.widget("time_3").value, 4.5);
    assert.equal(h.widget("reference_count").value, 3);
    h.disconnect("image_3");
    assert.deepEqual(h.imageInputs(), ["image_1", "image_2"]);
});

test("Remove clears both sources, while disconnect alone retains a selected file", async () => {
    const h = await createHarness();
    h.set("image_file_1", "a.png");
    h.connect("image_1");
    h.disconnect("image_1");
    assert.deepEqual(h.visibleRows(), [1, 2]);
    h.connect("image_1");
    h.remove(1);
    assert.equal(h.links.size, 0);
    assert.equal(h.widget("image_file_1").value, "(none)");
    assert.deepEqual(h.imageInputs(), ["image_1"]);
});

test("nine image limit, hole reuse, and complete removal", async () => {
    const h = await createHarness();
    for (let index = 1; index <= 9; index++) h.connect(`image_${index}`);
    assert.equal(h.imageInputs().length, 9);
    h.remove(2);
    assert.equal(h.imageInputs().length, 9, "Reuse the empty second entry as the spare");
    assert.equal(h.node.inputs.find((input) => input.name === "image_2").link, null);
    for (let index = 1; index <= 9; index++) h.remove(index);
    assert.deepEqual(h.imageInputs(), ["image_1"]);
    assert.equal(h.links.size, 0);
});

test("JSON hides rows and spare sockets while preserving connected image links", async () => {
    const h = await createHarness();
    const first = h.connect("image_1");
    h.set("reference_json", '[{"image":"a.png","time":0}]');
    assert.deepEqual(h.visibleRows(), []);
    assert.deepEqual(h.imageInputs(), ["image_1"]);
    assert.equal(h.node.inputs[h.links.get(first).target_slot].name, "image_1");
    h.set("reference_json", "");
    assert.deepEqual(h.imageInputs(), ["image_1", "image_2"]);
});

test("save/reload preserves socket order with holes and restores image/time values", async () => {
    const h = await createHarness();
    h.connect("image_1");
    h.connect("image_2");
    const third = h.connect("image_3");
    h.set("image_file_3", "c.png");
    h.set("time_3", 4.5);
    h.remove(1);
    const inputs = h.node.inputs.map((input) => ({ ...input }));
    const restored = new h.Node();
    restored.configure({ inputs, values: { image_file_3: "c.png", time_3: 4.5 } });
    h.flush();
    assert.deepEqual(restored.inputs.map((input) => input.name), inputs.map((input) => input.name));
    assert.equal(restored.inputs[h.links.get(third).target_slot].name, "image_3");
    assert.equal(h.widget("time_3", restored).value, 4.5);
    assert.equal(h.widget("image_file_3", restored).value, "c.png");
    assert.deepEqual(h.visibleRows(restored), [2, 3, 4]);
});

test("legacy workflows retain all filled rows and their connections", async () => {
    const h = await createHarness();
    const inputs = [
        { name: "clip", type: "CLIP", link: null },
        { name: "vae", type: "VAE", link: null },
        ...Array.from({ length: 9 }, (_, i) => ({ name: `image_${i + 1}`, type: "IMAGE", link: i === 2 ? 100 : null })),
    ];
    h.node.configure({ inputs, count: 4, values: { image_file_1: "a.png", image_file_4: "d.png", time_4: 5 } });
    h.flush();
    assert.deepEqual(h.visibleRows(), [1, 3, 4, 5]);
    assert.equal(h.widget("time_4").value, 5);
    assert.equal(h.node.inputs.find((input) => input.name === "image_3").link, 100);
    assert.equal(h.widget("reference_count").value, 4);
});
