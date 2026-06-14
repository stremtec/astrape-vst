class AstrapeCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const config = options.processorOptions || {};
    this.targetRate = config.targetRate || 16000;
    this.chunkSamples = config.chunkSamples || 80;
    this.ratio = sampleRate / this.targetRate;
    this.source = [];
    this.position = 0;
    this.output = [];
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel || !channel.length) return true;
    for (let index = 0; index < channel.length; index += 1) {
      this.source.push(channel[index]);
    }

    while (Math.ceil(this.position) < this.source.length) {
      const currentIndex = Math.ceil(this.position);
      const previousIndex = Math.max(0, currentIndex - 1);
      const fraction = this.position - previousIndex;
      const previous = this.source[previousIndex];
      const current = this.source[currentIndex];
      this.output.push(previous + (current - previous) * fraction);
      this.position += this.ratio;
      if (this.output.length === this.chunkSamples) {
        const block = new Float32Array(this.output);
        this.port.postMessage(block, [block.buffer]);
        this.output = [];
      }
    }

    const consumed = Math.max(0, Math.ceil(this.position) - 1);
    if (consumed > 0) {
      this.source.splice(0, consumed);
      this.position -= consumed;
    }
    return true;
  }
}

registerProcessor("astrape-capture", AstrapeCaptureProcessor);
