import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../web/js/h3_multiframe.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");
const persistent = (node) => node.widgets.filter((w) => w.serialize !== false && w.options?.serialize !== false);

async function createNode() {
    let extension;
    vm.runInNewContext(source, {
        app: { registerExtension(value) { extension = value; } },
        api: {}, URLSearchParams, queueMicrotask, Image: class {},
    });
    class Node {
        constructor() {
            this.size = [400, 600];
            this.inputs = Array.from({ length: 9 }, (_, i) => ({ name: `image_${i + 1}`, link: null }));
            this.widgets = [
                { name: "prompt", type: "text", value: "" },
                { name: "width", type: "number", value: 1344 },
                { name: "height", type: "number", value: 768 },
                { name: "duration", type: "number", value: 5 },
                { name: "reference_count", type: "number", value: 2 },
                { name: "ref_image_size", type: "combo", value: "match" },
                { name: "use_as_references", type: "toggle", value: true },
                { name: "reference_json", type: "text", value: "" },
            ];
            for (let index = 1; index <= 9; index++) {
                this.widgets.push(
                    { name: `image_file_${index}`, type: "combo", value: "(none)" },
                    { name: `time_${index}`, type: "number", value: (index - 1) * 1.5 },
                );
            }
            for (let index = 1; index <= 9; index++) {
                this.widgets.push({ name: `use_as_reference_${index}`, type: "combo", value: "default" });
            }
            this.onNodeCreated();
        }
        addWidget(type, name, value, callback, options) {
            const widget = { type, name, value, callback, options };
            this.widgets.push(widget);
            return widget;
        }
        addCustomWidget(widget) { this.widgets.push(widget); }
        addInput(name, type) { this.inputs.push({ name, type, link: null }); }
        removeInput(index) { this.inputs.splice(index, 1); }
        disconnectInput(index) { this.inputs[index].link = null; }
        configure(data) {
            // Restore by position to exercise compatibility with older frontends.
            persistent(this).forEach((widget, index) => {
                if (index < data.widgets_values.length) widget.value = data.widgets_values[index];
            });
            this.onConfigure();
        }
        computeSize() { return [400, 600]; }
        setSize(size) { this.size = size; }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Node, { name: "MAIH3Multiframe" });
    return new Node();
}

const widget = (node, name) => node.widgets.find((w) => w.name === name);

test("reference roles hide timestamps and follow dynamic row/JSON visibility", async () => {
    const node = await createNode();
    assert.equal(widget(node, "use_as_reference_1").hidden, false);
    assert.equal(widget(node, "use_as_reference_3").hidden, true);
    assert.equal(widget(node, "time_1").hidden, true);
    widget(node, "image_file_1").value = "a.png";
    widget(node, "image_file_1").callback();
    assert.equal(widget(node, "use_as_reference_2").hidden, false);
    widget(node, "use_as_reference_1").value = "no";
    widget(node, "use_as_reference_1").callback();
    assert.equal(widget(node, "time_1").hidden, false);
    widget(node, "use_as_reference_1").value = "yes";
    widget(node, "use_as_reference_1").callback();
    assert.equal(widget(node, "time_1").hidden, true);
    widget(node, "reference_json").value = '[{"image":"a.png","time":0}]';
    widget(node, "reference_json").callback();
    assert.equal(widget(node, "use_as_reference_1").hidden, true);
    assert.equal(node.inputs.length, 0);
});

test("legacy positional workflows preserve file, time, and global default", async () => {
    const node = await createNode();
    const old = ["Saved prompt", 960, 544, 6, 4, "max", false, ""];
    for (let index = 1; index <= 9; index++) old.push(`image${index}.png`, index / 2);
    node.configure({ widgets_values: old });
    await Promise.resolve();
    assert.equal(widget(node, "prompt").value, "Saved prompt");
    assert.equal(widget(node, "use_as_references").value, false);
    for (let index = 1; index <= 9; index++) {
        assert.equal(widget(node, `image_file_${index}`).value, `image${index}.png`);
        assert.equal(widget(node, `time_${index}`).value, index / 2);
        assert.equal(widget(node, `use_as_reference_${index}`).value, "default");
    }
});

test("named legacy workflows restore correctly when widgets are interleaved", async () => {
    const node = await createNode();
    const named = Object.fromEntries(persistent(node)
        .filter((w) => !w.name.startsWith("use_as_reference_"))
        .map((w) => [w.name, w.value]));
    named.image_file_2 = "second.png";
    named.time_2 = 4.25;
    named.use_as_references = false;
    node.configure({ widgets_values: Object.values(named), widgets_values_named: named });
    await Promise.resolve();
    assert.equal(widget(node, "image_file_2").value, "second.png");
    assert.equal(widget(node, "time_2").value, 4.25);
    assert.equal(widget(node, "use_as_references").value, false);
    assert.equal(widget(node, "use_as_reference_2").value, "default");
});

test("new named and positional workflows retain mixed reference selections", async () => {
    for (const named of [true, false]) {
        const saved = await createNode();
        widget(saved, "image_file_1").value = "first.png";
        widget(saved, "image_file_2").value = "second.png";
        widget(saved, "use_as_reference_1").value = "yes";
        widget(saved, "use_as_reference_2").value = "no";
        const data = { widgets_values: persistent(saved).map((w) => w.value) };
        if (named) data.widgets_values_named = Object.fromEntries(persistent(saved).map((w) => [w.name, w.value]));
        const restored = await createNode();
        restored.configure(data);
        await Promise.resolve();
        assert.equal(widget(restored, "image_file_2").value, "second.png");
        assert.equal(widget(restored, "use_as_reference_1").value, "yes");
        assert.equal(widget(restored, "use_as_reference_2").value, "no");
    }
});

test("default roles follow the global setting and explicit roles override it", async () => {
    const node = await createNode();
    widget(node, "use_as_references").value = false;
    widget(node, "use_as_references").callback();
    assert.equal(widget(node, "time_1").hidden, false);
    widget(node, "use_as_reference_1").value = "yes";
    widget(node, "use_as_reference_1").callback();
    assert.equal(widget(node, "time_1").hidden, true);
});

test("dynamic removal preserves the other pictures' roles and timestamps", async () => {
    const node = await createNode();
    for (let index = 1; index <= 3; index++) {
        widget(node, `image_file_${index}`).value = `image${index}.png`;
        widget(node, `image_file_${index}`).callback();
    }
    widget(node, "use_as_reference_2").value = "no";
    widget(node, "use_as_reference_2").callback();
    widget(node, "use_as_reference_3").value = "yes";
    widget(node, "use_as_reference_3").callback();
    widget(node, "time_3").value = 4.5;
    widget(node, "Remove image 2").callback();
    await Promise.resolve();
    assert.equal(widget(node, "image_file_2").value, "(none)");
    assert.equal(widget(node, "use_as_reference_2").hidden, true);
    assert.equal(widget(node, "image_file_3").value, "image3.png");
    assert.equal(widget(node, "use_as_reference_3").value, "yes");
    assert.equal(widget(node, "time_3").value, 4.5);
    assert.equal(widget(node, "time_3").hidden, true);
    assert.equal(widget(node, "use_as_reference_4").hidden, false);
});
