# 图标使用速查表

本 skill 强制使用 **Lucide Icons** 或 **Tabler Icons** 内联 SVG，禁止使用 Emoji。

## 推荐库

- **Lucide**: https://lucide.dev
  - 风格：简洁、圆润、现代
  - 安装：`npm install lucide` 或 `npm install lucide-react` / `lucide-vue-next`
- **Tabler**: https://tabler-icons.io
  - 风格：线条清晰、图标数量多
  - 安装：`npm install @tabler/icons-react` / `@tabler/icons-vue`

## 常用图标映射

| 场景 | Lucide | Tabler |
|------|--------|--------|
| 菜单 | `Menu` | `Menu2` |
| 关闭 | `X` | `X` |
| 首页 | `Home` | `Home` |
| 搜索 | `Search` | `Search` |
| 用户 | `User` | `User` |
| 设置 | `Settings` | `Settings` |
| 箭头右 | `ArrowRight` | `ArrowRight` |
| 箭头右上 | `ArrowUpRight` | `ArrowUpRight` |
| 下箭头 | `ChevronDown` | `ChevronDown` |
| 添加 | `Plus` | `Plus` |
| 删除 | `Trash2` | `Trash` |
| 编辑 | `Edit3` | `Edit` |
| 复制 | `Copy` | `Copy` |
| 成功 | `CheckCircle2` | `CircleCheck` |
| 警告 | `AlertCircle` | `AlertCircle` |
| 信息 | `Info` | `InfoCircle` |
| 错误 | `XCircle` | `CircleX` |
| 加载 | `Loader2` | `Loader2` |
| 功能/快速 | `Zap` | `Bolt` |
| 安全 | `Shield` | `Shield` |
| 全球 | `Globe` | `World` |
| 图层 | `Layers` | `Layers` |
| 闪亮/AI | `Sparkles` | `Sparkles` |
| 图表 | `BarChart3` | `ChartBar` |

## 内联 SVG 规范

```html
<svg
  xmlns="http://www.w3.org/2000/svg"
  width="24"
  height="24"
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="1.5"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"
>
  <path d="M4 12h16M14 6l6 6-6 6" />
</svg>
```

## React 示例

```jsx
import { ArrowRight, Sparkles } from 'lucide-react';

<button className="cta">
  开始使用 <ArrowRight size={20} strokeWidth={1.5} />
</button>
```

## 无障碍

- 装饰性图标：`aria-hidden="true"`
- 图标按钮：`<button aria-label="关闭"><X ... /></button>`
- 避免单独使用图标表达关键状态，必要时配合文字