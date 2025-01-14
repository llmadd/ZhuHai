'use client'

import { memo } from 'react'
import ReactMarkdown, { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypePrism from 'rehype-prism-plus'
import { CodeBlock } from './code-block'
import { useLocale } from '@/contexts/locale-context'
import { generateId } from '@/lib/toc'

interface CustomMarkdownProps {
    children: string
}

function CustomMarkdownComponent({ children }: CustomMarkdownProps) {
    const { locale } = useLocale()

    const createHeading = (level: 1 | 2 | 3 | 4 | 5 | 6) => {
        const Component = `h${level}` as keyof JSX.IntrinsicElements
        return ({ children, ...props }: any) => {
            const id = generateId(String(children))
            return (
                <Component id={id} {...props}>
                    {children}
                </Component>
            )
        }
    }

    const components: Partial<Components> = {
        code: ({ className, children, ...props }) => {
            // 提取代码内容的函数
            const extractTextContent = (child: any): string => {
                if (typeof child === 'string') return child
                if (!child?.props?.children) return ''

                if (Array.isArray(child.props.children)) {
                    return child.props.children.map(extractTextContent).join('')
                }

                return extractTextContent(child.props.children)
            }

            // 处理代码内容
            let content = ''
            if (Array.isArray(children)) {
                content = children.map(child => extractTextContent(child)).join('')
            } else {
                content = extractTextContent(children)
            }

            // 清理内容
            const cleanContent = content.replace(/\n$/, '')

            if (!className) {
                return <code className="font-mono text-sm" {...props}>{cleanContent}</code>
            }

            return <CodeBlock className={className}>{cleanContent}</CodeBlock>
        },
        h1: createHeading(1),
        h2: createHeading(2),
        h3: createHeading(3),
        h4: createHeading(4),
        h5: createHeading(5),
        h6: createHeading(6),
        img: ({ src, alt, ...props }) => {
            if (!src) return null

            return (
                <img
                    src={src}
                    alt={alt || ''}
                    className="max-w-full h-auto rounded-lg my-4"
                    loading="lazy"
                    {...props}
                />
            )
        },
    }

    return (
        <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypePrism]}
            components={components}
            key={locale}
        >
            {children}
        </ReactMarkdown>
    )
}

export const CustomMarkdown = memo(CustomMarkdownComponent)