import { remark } from 'remark'
import { visit } from 'unist-util-visit'
import { toString } from 'mdast-util-to-string'
import slugify from 'slugify'

export async function getTableOfContents(content: string) {
    const tree = await remark().parse(content)
    const headings: { id: string; text: string; level: number }[] = []
    const slugs = new Set<string>()

    visit(tree, 'heading', (node: any) => {
        const text = toString(node)
        let id = slugify(text, { lower: true })

        // 如果 id 已存在，添加数字后缀
        let counter = 1
        let uniqueId = id
        while (slugs.has(uniqueId)) {
            uniqueId = `${id}-${counter}`
            counter++
        }

        slugs.add(uniqueId)
        headings.push({
            id: uniqueId,
            text,
            level: node.depth,
        })
    })

    return headings
} 