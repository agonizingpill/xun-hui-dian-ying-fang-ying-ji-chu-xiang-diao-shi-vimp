import {awaitCompute,request} from './api.js';

export const studio = {
  board: () => request('/booths'),
  outlet: (booth, body) => request('/booths/' + booth + '/outlet', body),
  reports: booth => request('/optics/' + booth),
  test: (booth, body) => request('/optics/' + booth + '/evaluate', body),
  result: awaitCompute,
  save: body => request('/optics/save', body),
  cabinet: () => request('/lenses'),
  lend: body => request('/lenses/lend', body),
  returnLens: (code, body) => request('/lenses/' + code + '/return', body),
  clear: (booth, body) => request('/booths/' + booth + '/clear',body),
  models: () => request('/models'),
  addSamples: body => request('/models/readings', body),
  trainModel: body => request('/models/train', body),
  activateModel: version => request('/models/' + encodeURIComponent(version) + '/activate', {}),
  withdrawModel: version => request('/models/' + encodeURIComponent(version) + '/withdraw', {}),
};
