import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "ASCIINovelTextArt";

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) || null;
}

function setWidgetValue(widget, value) {
  if (!widget) {
    return;
  }
  widget.value = value;
  widget.callback?.(value);
}

async function uploadTextFile(file) {
  const formData = new FormData();
  formData.append("file", file, file.name);

  const response = await fetch(api.apiURL("/asci/upload-text"), {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload?.error || "Failed to upload text file.");
  }

  return payload.filename;
}

function createUploadElement(node) {
  const wrapper = document.createElement("div");
  wrapper.style.display = "flex";
  wrapper.style.flexDirection = "column";
  wrapper.style.gap = "6px";
  wrapper.style.padding = "8px";
  wrapper.style.borderRadius = "8px";
  wrapper.style.border = "1px dashed rgba(255, 255, 255, 0.28)";
  wrapper.style.background = "rgba(255, 255, 255, 0.04)";
  wrapper.style.minHeight = "78px";
  wrapper.style.boxSizing = "border-box";

  const title = document.createElement("div");
  title.textContent = "TXT Upload";
  title.style.fontSize = "12px";
  title.style.fontWeight = "600";
  title.style.color = "var(--input-text, #ddd)";

  const hint = document.createElement("div");
  hint.textContent = "Drop a .txt file here or choose one.";
  hint.style.fontSize = "12px";
  hint.style.lineHeight = "1.35";
  hint.style.color = "var(--descrip-text, #aaa)";

  const controls = document.createElement("div");
  controls.style.display = "flex";
  controls.style.alignItems = "center";
  controls.style.gap = "8px";

  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "Choose .txt";
  button.style.padding = "4px 10px";
  button.style.borderRadius = "6px";
  button.style.border = "1px solid rgba(255, 255, 255, 0.18)";
  button.style.background = "rgba(255, 255, 255, 0.08)";
  button.style.color = "var(--input-text, #ddd)";
  button.style.cursor = "pointer";

  const status = document.createElement("div");
  status.style.fontSize = "12px";
  status.style.color = "var(--input-text, #ddd)";
  status.style.overflow = "hidden";
  status.style.textOverflow = "ellipsis";
  status.style.whiteSpace = "nowrap";
  status.textContent = "No file uploaded.";

  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".txt,text/plain";
  input.style.display = "none";

  controls.append(button, status);
  wrapper.append(title, hint, controls, input);

  const setBusy = (busy) => {
    button.disabled = busy;
    wrapper.style.opacity = busy ? "0.7" : "1";
  };

  const syncStatus = () => {
    const pathWidget = getWidget(node, "text_file_path");
    const value = pathWidget?.value;
    status.textContent = value ? `Uploaded: ${value}` : "No file uploaded.";
  };

  const handleFile = async (file) => {
    if (!file) {
      return;
    }
    if (!file.name.toLowerCase().endsWith(".txt")) {
      status.textContent = "Only .txt files are supported.";
      return;
    }

    setBusy(true);
    status.textContent = `Uploading: ${file.name}`;

    try {
      const uploadedName = await uploadTextFile(file);
      setWidgetValue(getWidget(node, "text_file_path"), uploadedName);
      status.textContent = `Uploaded: ${uploadedName}`;
    } catch (error) {
      status.textContent = error instanceof Error ? error.message : "Upload failed.";
    } finally {
      input.value = "";
      setBusy(false);
      node.setSize(node.computeSize());
    }
  };

  button.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    void handleFile(input.files?.[0] || null);
  });

  wrapper.addEventListener("dragover", (event) => {
    event.preventDefault();
    wrapper.style.borderColor = "rgba(255, 255, 255, 0.5)";
  });
  wrapper.addEventListener("dragleave", () => {
    wrapper.style.borderColor = "rgba(255, 255, 255, 0.28)";
  });
  wrapper.addEventListener("drop", (event) => {
    event.preventDefault();
    wrapper.style.borderColor = "rgba(255, 255, 255, 0.28)";
    void handleFile(event.dataTransfer?.files?.[0] || null);
  });

  return {
    element: wrapper,
    syncStatus,
    openPicker() {
      input.click();
    },
  };
}

function wrapPathWidget(node, syncStatus) {
  const widget = getWidget(node, "text_file_path");
  if (!widget || widget.__asciiNovelWrapped) {
    syncStatus();
    return;
  }

  const originalCallback = widget.callback;
  widget.callback = function callback(value, ...args) {
    const result = originalCallback?.call(this, value, ...args);
    syncStatus();
    return result;
  };
  widget.__asciiNovelWrapped = true;
  syncStatus();
}

function setupNovelNode(node) {
  if (node.__asciiNovelUploadUi) {
    wrapPathWidget(node, node.__asciiNovelUploadUi.syncStatus);
    return;
  }

  const uploadUi = createUploadElement(node);
  node.__asciiNovelUploadUi = uploadUi;

  if (typeof node.addDOMWidget === "function") {
    const widget = node.addDOMWidget("txt_upload", "txt_upload", uploadUi.element, {
      serialize: false,
      hideOnZoom: false,
    });
    if (widget) {
      widget.serializeValue = () => undefined;
      widget.computeSize = (width) => [width, 94];
    }
  } else {
    const button = node.addWidget("button", "txt_upload", "Choose .txt", () => {
      uploadUi.openPicker();
    }, {
      serialize: false,
    });
    if (button) {
      button.options = {
        ...(button.options || {}),
        serialize: false,
      };
    }
  }

  wrapPathWidget(node, uploadUi.syncStatus);
  node.setSize(node.computeSize());
}

app.registerExtension({
  name: "comfyui-color-ascii-art-node.ascii-novel-text-art",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE && nodeData.display_name !== "ASCII Novel Text Art") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onNodeCreatedWrapped(...args) {
      const result = onNodeCreated?.apply(this, args);
      setupNovelNode(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function onConfigureWrapped(...args) {
      const result = onConfigure?.apply(this, args);
      setupNovelNode(this);
      return result;
    };
  },
});
