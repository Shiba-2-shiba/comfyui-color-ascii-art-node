import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "ASCIITextLayoutPlanner";
const INFO_WIDGETS = [
  ["planner_image_size", "image_size", "Image Size"],
  ["planner_grid", "grid", "Grid"],
  ["planner_required_chars", "required_chars", "Required Chars"],
  ["planner_render_size", "render_size", "Render Size"],
];

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

function ensureInfoWidgets(node) {
  node.__asciiPlannerState = node.__asciiPlannerState || {
    imageWidth: null,
    imageHeight: null,
    requestToken: 0,
  };

  for (const [name, key, label] of INFO_WIDGETS) {
    const existingWidget = getWidget(node, name);
    if (existingWidget) {
      node.__asciiPlannerState[key] = existingWidget;
      continue;
    }
    const widget = node.addWidget("text", name, `${label}: -`, null, {
      serialize: false,
      disabled: true,
    });
    widget.options = {
      ...(widget.options || {}),
      disabled: true,
      serialize: false,
    };
    node.__asciiPlannerState[key] = widget;
  }

  node.setSize(node.computeSize());
}

function updatePlannerInfo(node) {
  const state = node.__asciiPlannerState;
  if (!state) {
    return;
  }

  const pixelSize = Number(getWidget(node, "pixel_size")?.value ?? 0);
  const aspectRatio = Number(getWidget(node, "aspect_ratio_correction")?.value ?? 0);
  const resolutionScale = Number(getWidget(node, "resolution_scale")?.value ?? 0);

  if (!state.imageWidth || !state.imageHeight || pixelSize <= 0 || aspectRatio <= 0 || resolutionScale <= 0) {
    setWidgetValue(state.image_size, "Image Size: -");
    setWidgetValue(state.grid, "Grid: -");
    setWidgetValue(state.required_chars, "Required Chars: -");
    setWidgetValue(state.render_size, "Render Size: -");
    return;
  }

  const gridWidth = Math.max(1, Math.floor(state.imageWidth / pixelSize));
  const gridHeight = Math.max(1, Math.floor(state.imageHeight / (pixelSize * aspectRatio)));
  const requiredChars = gridWidth * gridHeight;
  const renderWidth = Math.max(1, Math.floor(state.imageWidth * resolutionScale));
  const renderHeight = Math.max(1, Math.floor(state.imageHeight * resolutionScale));

  setWidgetValue(state.image_size, `Image Size: ${state.imageWidth}x${state.imageHeight}`);
  setWidgetValue(state.grid, `Grid: ${gridWidth}x${gridHeight}`);
  setWidgetValue(state.required_chars, `Required Chars: ${requiredChars}`);
  setWidgetValue(state.render_size, `Render Size: ${renderWidth}x${renderHeight}`);
}

async function loadImageDimensions(node) {
  const state = node.__asciiPlannerState;
  const imageName = getWidget(node, "image")?.value;
  const token = (state.requestToken || 0) + 1;
  state.requestToken = token;

  if (!imageName) {
    state.imageWidth = null;
    state.imageHeight = null;
    updatePlannerInfo(node);
    return;
  }

  try {
    const params = new URLSearchParams({
      filename: imageName,
      type: "input",
      rand: String(Date.now()),
    });
    const imageUrl = api.apiURL(`/view?${params.toString()}`);
    const dimensions = await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
      image.onerror = () => reject(new Error(`Failed to load image dimensions for ${imageName}`));
      image.src = imageUrl;
    });

    if (state.requestToken !== token) {
      return;
    }

    state.imageWidth = dimensions.width;
    state.imageHeight = dimensions.height;
    updatePlannerInfo(node);
  } catch (_error) {
    if (state.requestToken !== token) {
      return;
    }
    state.imageWidth = null;
    state.imageHeight = null;
    updatePlannerInfo(node);
  }
}

function wrapWidgetCallback(node, widgetName, handler) {
  const widget = getWidget(node, widgetName);
  if (!widget || widget.__asciiPlannerWrapped) {
    return;
  }

  const originalCallback = widget.callback;
  widget.callback = function callback(value, ...args) {
    const result = originalCallback?.call(this, value, ...args);
    handler(node, value);
    return result;
  };
  widget.__asciiPlannerWrapped = true;
}

function setupPlannerNode(node) {
  ensureInfoWidgets(node);

  wrapWidgetCallback(node, "image", (currentNode) => {
    void loadImageDimensions(currentNode);
  });
  for (const widgetName of ["pixel_size", "aspect_ratio_correction", "resolution_scale"]) {
    wrapWidgetCallback(node, widgetName, updatePlannerInfo);
  }

  void loadImageDimensions(node);
}

app.registerExtension({
  name: "comfyui-color-ascii-art-node.ascii-text-layout-planner",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE && nodeData.display_name !== "ASCII Text Layout Planner") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onNodeCreatedWrapped(...args) {
      const result = onNodeCreated?.apply(this, args);
      setupPlannerNode(this);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function onConfigureWrapped(...args) {
      const result = onConfigure?.apply(this, args);
      setupPlannerNode(this);
      return result;
    };
  },
});
