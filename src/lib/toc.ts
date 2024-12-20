import { remark } from 'remark'
import { visit } from 'unist-util-visit'
import { toString } from 'mdast-util-to-string'
import slugify from 'slugify'
import { Node } from 'unist'

interface HeadingNode extends Node {
    depth: number
}

// 创建一个统一的 ID 生成函数
export function generateId(text: string) {
    // 首先移除 Markdown 标记符号
    const cleanText = text
        .replace(/^#+\s+/, '')  // 移除标题的 # 符号
        .replace(/`/g, '')      // 移除代码块符号
        .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')  // 处理链接
        .trim()

    // 如果文本以数字或特殊字符开头，添加前缀
    const prefix = /^[^a-zA-Z]/.test(cleanText) ? 'section-' : ''

    const slug = slugify(cleanText, {
        lower: true,
        strict: true,
        trim: true,
        replacement: '-',
        remove: /[*+~.()'"!:@#]/g
    })

    // 如果 slug 为空或只包含连字符，使用固定前缀
    if (!slug || slug === '-') {
        return 'section-untitled'
    }

    return `${prefix}${slug}`
}

export async function getTableOfContents(content: string) {
    const tree = await remark().parse(content)
    const headings: { id: string; text: string; level: number }[] = []
    const usedIds = new Set<string>()

    visit(tree, 'heading', (node: HeadingNode) => {
        const text = toString(node)
        const cleanText = text
            .replace(/^#+\s+/, '')
            .replace(/`/g, '')
            .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
            .trim()

        // 如果文本以数字或特殊字符开头，添加前缀
        const prefix = /^[^a-zA-Z]/.test(cleanText) ? 'section-' : ''
        let baseId = slugify(cleanText, {
            lower: true,
            strict: true,
            trim: true,
            replacement: '-',
            remove: /[*+~.()'"!:@#]/g
        })

        // 如果 baseId 为空或只包含连字符，使用固定前缀
        if (!baseId || baseId === '-') {
            baseId = 'section-untitled'
        } else {
            baseId = `${prefix}${baseId}`
        }

        let id = baseId
        let counter = 1
        while (usedIds.has(id)) {
            id = `${baseId}-${counter}`
            counter++
        }

        usedIds.add(id)
        headings.push({
            id,
            text,
            level: node.depth,
        })
    })

    return headings
} 