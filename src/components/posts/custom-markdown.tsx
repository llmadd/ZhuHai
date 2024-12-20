'use client'

import { FC, memo } from 'react'
import ReactMarkdown, { Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypePrism from 'rehype-prism-plus'
import { CodeBlock } from './code-block'
import { useLocale } from '@/contexts/locale-context'
import { generateId } from '@/lib/toc'

interface CustomMarkdownProps {
    children: string
}

const CustomMarkdownComponent: FC<CustomMarkdownProps> = ({ children }) => {
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
            if (!className) {
                return <code className="font-mono text-sm" {...props}>{children}</code>
            }
            return <CodeBlock className={className}>{String(children)}</CodeBlock>
        },
        h1: createHeading(1),
        h2: createHeading(2),
        h3: createHeading(3),
        h4: createHeading(4),
        h5: createHeading(5),
        h6: createHeading(6),
        img: () => null
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

CustomMarkdownComponent.displayName = 'CustomMarkdown'

export const CustomMarkdown = memo(CustomMarkdownComponent) 