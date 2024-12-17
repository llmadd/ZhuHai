interface CategoryNames {
    zh: string
    en: string
}

export const categoryMap: Record<string, CategoryNames> = {
    '教程': {
        zh: '教程分享',
        en: 'Tutorials'
    },
    '随笔': {
        zh: '生活随笔',
        en: 'Life Notes'
    },
    '小问题': {
        zh: '问题解决',
        en: 'Problem Solving'
    },
    '小应用': {
        zh: '小应用',
        en: 'Mini Apps'
    }
}

export type CategoryKey = keyof typeof categoryMap