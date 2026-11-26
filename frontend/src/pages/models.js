import {el} from '../dom.js';
import {studio} from '../booth-api.js';
import {artifacts, fields} from '../booth-state.js';
import {input, select, submit} from '../inputs.js';
import {status, titleBlock} from '../widgets.js';
import {tag} from '../components/status-tag.js';

const modelStatuses = {
  candidate: '候选版',
  active: '启用中',
  retired: '已替下',
  withdrawn: '已撤回',
};

function pct(value) {
  return value === undefined || value === null ? '—' : (value * 100).toFixed(1) + '%';
}

function gateLine(model) {
  const gate = model.gate;
  if (!gate || !gate.checks) return status('尚未生成留出比较。');
  return el('div', {class: 'gate-grid'}, Object.entries(gate.checks).map(([label, item]) =>
    el('span', {class: item.regressed ? 'gate bad' : 'gate good'},
      artifacts[label] || label,
      ' 漏判 ', item.active, '→', item.candidate,
      item.regressed ? '（变多）' : '（未增加）')));
}

function versionCard(model, role, activeVersion, onChanged) {
  const active = model.version === activeVersion;
  const canReview = role === 'reviewer';
  const actions = [];
  if (canReview && model.status === 'candidate') {
    actions.push(submit('activate-' + model.version, [], '启用新版',
      async () => {
        await studio.activateModel(model.version);
        await onChanged(model.version + ' 已启用；各工位新单一次性切换。');
      }));
  }
  if (canReview && active && model.version !== 'ext-20261124-demo') {
    actions.push(submit('withdraw-' + model.version, [], '撤回新版',
      async () => {
        const result = await studio.withdrawModel(model.version);
        await onChanged('已撤回；新单恢复为 ' + result.active_version + '。');
      }));
  }
  return el('article', {class: 'record model-card', 'data-version': model.version},
    el('h3', {class: 'mono'}, model.version),
    el('p', {}, tag(modelStatuses, model.status), active ? tag({yes: '当前接单版'}, 'yes') : null),
    el('p', {}, '训练 ', model.training_count, ' 条（新读数 ', model.new_sample_count,
      '） · 同一留出集 ', model.holdout_count, ' 条 · 准确率 ', pct(model.metrics?.accuracy)),
    gateLine(model),
    model.replaced_version ? el('p', {class: 'mono'}, '启用时替下：' + model.replaced_version) : null,
    el('details', {}, el('summary', {}, '混淆矩阵与版号说明'),
      el('p', {class: 'mono'}, '标签顺序：' + (model.metrics?.labels || []).join(' / ')),
      el('pre', {class: 'matrix'}, JSON.stringify(model.metrics?.matrix || [], null, 2))),
    ...actions);
}

export async function models(root, role) {
  const board = el('section', {class: 'ledger'});
  const editor = el('section', {class: 'action'});
  const notice = el('div', {});

  async function load(message = '') {
    const data = await studio.models();
    if (!board.isConnected) return;
    notice.replaceChildren(message ? status(message, 'ok') : null);
    board.replaceChildren(
      el('h2', {}, '版本台账'),
      el('div', {class: 'callout'}, el('strong', {}, '当前接单版：'),
        el('span', {class: 'mono'}, data.active_version),
        el('p', {}, '启用只影响之后受理的新判别单；已排队任务按接单时版号计算，中断恢复也不变。')),
      data.models.length ? el('div', {class: 'film model-list'},
        data.models.map(model => versionCard(model, role, data.active_version, load)))
        : status('还没有模型版本。')
    );
  }

  editor.replaceChildren(
    el('h2', {}, '带标签的新读数'),
    role === 'reviewer'
      ? el('div', {},
          submit('add-training-reading', [
            el('div', {class: 'fields'}, Object.entries(fields).map(([key, label]) =>
              input(key, label + ' / 归一值', {
                type: 'number', min: '0', max: '1', step: '.01',
                value: ({edge_loss: '.10', color_error: '.11',
                         flicker_index: '.10', frame_var: '.09'})[key],
              }))),
            select('artifact', '标签 / 复核标签', Object.entries(artifacts)),
            input('code', '读数/原检测单号（可选）', {placeholder: 'RD-1124-SOFT', required: undefined}),
          ], '加入带标签读数', async form => {
            const body = {
              artifact: form.get('artifact'),
              code: form.get('code') || undefined,
              edge_loss: Number(form.get('edge_loss')),
              color_error: Number(form.get('color_error')),
              flicker_index: Number(form.get('flicker_index')),
              frame_var: Number(form.get('frame_var')),
            };
            const result = await studio.addSamples(body);
            await load('已加入 ' + result.created + ' 条带标签读数。');
          }),
          el('h2', {}, '训练候选版 · 训练画质模型'),
          submit('train-model', [
            input('version', '指定版号（可选）', {placeholder: 'cand-tour-26', required: undefined}),
            input('note', '版本说明（可选）', {placeholder: '巡回途中追加复核标签', required: undefined}),
            el('label', {}, el('input', {type: 'checkbox', name: 'include_baseline', checked: 'checked'}),
              el('span', {}, '合并内置 220 条训练样本')),
          ], '训练候选版', async form => {
            const result = await studio.trainModel({
              version: form.get('version') || undefined,
              note: form.get('note') || undefined,
              include_baseline: form.get('include_baseline') === 'on',
            });
            await load(result.version + ' 已训练；门槛：' + (result.gate?.eligible ? '可启用' : '有异常类漏判增加'));
          }))
      : status('只有复核员可以录入标签、训练和换版。')
  );

  root.replaceChildren(
    titleBlock('巡回放映 / 模型', '画质模型训练与换版',
      '新读数带复核标签后可训练候选版。候选版与启用时在用版使用同一套 60 条固定留出样本；任一异常类漏判变多，后端会拒绝启用。'),
    notice,
    el('div', {class: 'spread reversed'}, board, editor)
  );
  await load().catch(error => board.replaceChildren(status(error.message, 'error')));
}
