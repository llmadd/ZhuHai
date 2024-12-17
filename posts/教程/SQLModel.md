---
title: 
    zh: 'SQLModel 教程'
    en: 'SQLModel Tutorial'
date: '2024-12-06'
author: 'Hai'
coverImage: 'https://sqlmodel.tiangolo.com/img/logo-margin/logo-margin-vector.svg#only-light'
coverImageAlt:
    zh: 'SQLModel 图标'
    en: 'SQLModel Icon'
tags: ['SQLModel', 'python', 'ORM']
status: 'published'
---

<!-- Chinese Content -->

# SQLModel 介绍

SQLModel 是一个ORM库，用于通过 Python 代码与 Python 对象中的 SQL 数据库进行交互。它的设计直观、易于使用、高度兼容且功能强大。SQLModel 基于 Python 类型注释，并由 [Pydantic](https://pydantic-docs.helpmanual.io/) 和 [SQLAlchemy](https://sqlalchemy.org/) 提供支持。

---

# SQLModel - 初始化引擎与表操作

本节将详细介绍如何使用 SQLModel 初始化数据库引擎、创建表和删除表的操作。每个部分都会附带完整的代码示例和详细讲解，帮助您快速上手。

---

## 1. 初始化数据库引擎 (Initialize the Database Engine)

在使用 SQLModel 与数据库交互之前，必须先初始化数据库引擎。SQLModel 支持多种数据库类型（如 SQLite、PostgreSQL、MySQL 等），我们以 SQLite 为例。

### 示例代码

```python
from sqlmodel import create_engine

# 初始化 SQLite 数据库引擎
engine = create_engine("sqlite:///database.db", echo=True)
```

### 代码详解

1. **`create_engine`**：
   - 这是 SQLModel 提供的函数，用于创建数据库引擎。
   - 参数 `"sqlite:///database.db"` 指定了数据库类型和路径：
     - `sqlite://` 表示使用 SQLite 数据库。
     - `database.db` 是数据库文件的名称。如果文件不存在，SQLite 会自动创建。

2. **`echo=True`**：
   - 设置为 `True` 时，SQLModel 会在控制台输出所有的 SQL 语句，方便调试。

### SQLite 数据库路径说明

- **相对路径**：`sqlite:///database.db` 表示在当前目录下创建 `database.db` 文件。
- **绝对路径**：`sqlite:////full/path/to/database.db` 表示在指定路径创建数据库文件。

---

## 2. 创建表 (Create Tables)

SQLModel 提供了非常简单的方式来定义和创建数据库表。通过继承 `SQLModel`，我们可以定义表结构，然后使用 `SQLModel.metadata.create_all(engine)` 方法将这些表创建到数据库中。

### 示例代码

```python
from sqlmodel import SQLModel, Field

# 定义一个 User 表
class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)  # 主键
    name: str                                        # 用户名
    age: int                                         # 年龄

# 创建表
SQLModel.metadata.create_all(engine)
print("表已创建！")
```

### 代码详解

1. **定义模型**：
   - `class User(SQLModel, table=True)`：
     - 创建一个名为 `User` 的表。
     - `table=True` 表示这是一个数据库表，而不是普通的 Pydantic 模型。
   - `id: int = Field(default=None, primary_key=True)`：
     - 定义一个主键字段，类型为 `int`。
     - `default=None` 表示该字段可以为空，在插入数据时会自动生成。
   - `name: str` 和 `age: int`：
     - 定义表的其他字段，分别为字符串类型和整数类型。

2. **创建表**：
   - `SQLModel.metadata.create_all(engine)`：
     - 根据定义的模型，自动在数据库中创建对应的表。

3. **控制台输出**：
   - 如果 `echo=True`，控制台会打印 SQLModel 执行的 SQL 语句，例如：
     ```sql
     CREATE TABLE user (
         id INTEGER NOT NULL, 
         name VARCHAR NOT NULL, 
         age INTEGER NOT NULL, 
         PRIMARY KEY (id)
     );
     ```

---

## 3. 删除表 (Drop Tables)

如果需要删除数据库中的表，可以使用 `SQLModel.metadata.drop_all(engine)` 方法。注意，此操作会删除所有与 `SQLModel` 定义相关的表，请谨慎操作。

### 示例代码

```python
from sqlmodel import SQLModel

# 删除表
SQLModel.metadata.drop_all(engine)
print("表已删除！")
```

### 代码详解

1. **删除表**：
   - `SQLModel.metadata.drop_all(engine)`：
     - 删除与当前 `SQLModel` 定义相关的所有表。

2. **注意事项**：
   - 删除表后，表中的所有数据也会被清空。
   - 如果只想删除某个特定表，可以考虑直接使用 SQLAlchemy 提供的低级方法。

---

## 4. 创建表和删除表的完整案例

以下是一个完整的案例，展示如何定义表、创建表、插入数据、删除表的整个流程。

### 示例代码

```python
from sqlmodel import SQLModel, Field, create_engine

# 初始化数据库引擎
engine = create_engine("sqlite:///example.db", echo=True)

# 定义一个 User 表
class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)  # 主键
    name: str                                        # 用户名
    age: int                                         # 年龄

# 创建表
print("正在创建表...")
SQLModel.metadata.create_all(engine)
print("表已创建！")

# 插入数据（仅为演示，不涉及 Session 的详细内容）
from sqlmodel import Session

with Session(engine) as session:
    user = User(name="Alice", age=25)
    session.add(user)
    session.commit()
    print("数据已插入！")

# 删除表
print("正在删除表...")
SQLModel.metadata.drop_all(engine)
print("表已删除！")
```

### 运行结果

1. 数据库引擎初始化时，会输出类似以下信息：
   ```
   2024-12-06 11:00:00 INFO sqlalchemy.engine.Engine BEGIN (implicit)
   2024-12-06 11:00:00 INFO sqlalchemy.engine.Engine PRAGMA main.table_info("user")
   ```

2. 创建表时，会输出生成的 SQL 语句：
   ```sql
   CREATE TABLE user (
       id INTEGER NOT NULL, 
       name VARCHAR NOT NULL, 
       age INTEGER NOT NULL, 
       PRIMARY KEY (id)
   );
   ```

3. 删除表时，会输出删除表的 SQL 语句：
   ```sql
   DROP TABLE user;
   ```

---

## 5. 小结

- **初始化引擎**：通过 `create_engine` 创建数据库引擎，支持多种数据库类型。
- **创建表**：使用 `SQLModel.metadata.create_all(engine)` 方法创建表。
- **删除表**：使用 `SQLModel.metadata.drop_all(engine)` 方法删除表。


---

# SQLModel - 对表的增删改查操作

在本节中，我们将详细讲解如何对表进行数据的增（插入数据）、删（删除数据）、改（更新数据）、查（查询数据）操作。每个部分都包含完整的代码示例和详细的讲解，帮助您快速掌握 SQLModel 的基本数据操作。

---

## 1. 插入数据 (Insert Data)

在 SQLModel 中，可以通过 `Session` 对数据库表插入数据。以下是一个完整的插入数据示例。

### 示例代码

```python
from sqlmodel import SQLModel, Field, create_engine, Session

# 定义数据库引擎
engine = create_engine("sqlite:///example.db", echo=True)

# 定义 User 表
class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)  # 主键
    name: str                                        # 用户名
    age: int                                         # 年龄

# 创建表
SQLModel.metadata.create_all(engine)

# 插入数据
with Session(engine) as session:
    user1 = User(name="Alice", age=25)
    user2 = User(name="Bob", age=30)
    session.add(user1)  # 添加单个对象
    session.add(user2)  # 添加另一个对象
    session.commit()    # 提交事务
    print("数据已插入！")
```

### 代码详解

1. **创建数据对象**：
   - `User(name="Alice", age=25)`：创建一个 `User` 对象，表示一条记录。
   - `User(name="Bob", age=30)`：创建第二条记录。

2. **添加数据到会话**：
   - `session.add(user1)`：将 `user1` 添加到当前会话。
   - `session.add(user2)`：将 `user2` 添加到当前会话。

3. **提交事务**：
   - `session.commit()`：将会话中的所有更改提交到数据库。

4. **控制台输出**：
   - 如果 `echo=True`，控制台会输出插入数据的 SQL 语句，例如：
     ```sql
     INSERT INTO user (name, age) VALUES (?, ?)
     ```

---

## 2. 查询数据 (Query Data)

查询数据是数据库操作中最常见的操作之一。SQLModel 支持多种查询方式，包括查询所有数据、条件查询、分页查询等。

### 示例代码

```python
# 查询数据
with Session(engine) as session:
    # 查询所有用户
    users = session.query(User).all()
    print("所有用户：")
    for user in users:
        print(f"ID: {user.id}, Name: {user.name}, Age: {user.age}")

    # 条件查询：查找年龄大于 25 的用户
    users_over_25 = session.query(User).filter(User.age > 25).all()
    print("\n年龄大于 25 的用户：")
    for user in users_over_25:
        print(f"ID: {user.id}, Name: {user.name}, Age: {user.age}")
```

### 代码详解

1. **查询所有数据**：
   - `session.query(User).all()`：查询 `User` 表中的所有数据，返回一个列表。

2. **条件查询**：
   - `session.query(User).filter(User.age > 25).all()`：
     - 使用 `filter` 方法添加查询条件，查找年龄大于 25 的用户。
     - 返回符合条件的数据列表。

3. **遍历结果**：
   - 使用 `for` 循环遍历查询结果，并打印每条记录的字段值。

4. **控制台输出**：
   - 查询所有数据的 SQL 语句：
     ```sql
     SELECT user.id, user.name, user.age FROM user
     ```
   - 条件查询的 SQL 语句：
     ```sql
     SELECT user.id, user.name, user.age FROM user WHERE user.age > ?
     ```

---

## 3. 更新数据 (Update Data)

更新数据时，可以查询出需要修改的记录，修改其字段值后提交事务。

### 示例代码

```python
# 更新数据
with Session(engine) as session:
    # 查询需要更新的用户
    user_to_update = session.query(User).filter(User.name == "Alice").first()
    if user_to_update:
        user_to_update.age = 26  # 修改年龄
        session.add(user_to_update)  # 将修改后的对象添加到会话
        session.commit()  # 提交事务
        print(f"用户 {user_to_update.name} 的年龄已更新为 {user_to_update.age}")
```

### 代码详解

1. **查询需要更新的记录**：
   - `session.query(User).filter(User.name == "Alice").first()`：
     - 使用 `filter` 方法查找 `name` 为 `"Alice"` 的用户。
     - 使用 `first()` 方法返回第一条匹配的记录。

2. **修改字段值**：
   - `user_to_update.age = 26`：修改用户的 `age` 字段值。

3. **提交事务**：
   - `session.add(user_to_update)`：将修改后的对象添加到会话。
   - `session.commit()`：提交事务，将更改保存到数据库。

4. **控制台输出**：
   - 更新数据的 SQL 语句：
     ```sql
     UPDATE user SET age=? WHERE user.id = ?
     ```

---

## 4. 删除数据 (Delete Data)

删除数据时，可以查询出需要删除的记录，然后将其从会话中删除并提交事务。

### 示例代码

```python
# 删除数据
with Session(engine) as session:
    # 查询需要删除的用户
    user_to_delete = session.query(User).filter(User.name == "Bob").first()
    if user_to_delete:
        session.delete(user_to_delete)  # 从会话中删除对象
        session.commit()  # 提交事务
        print(f"用户 {user_to_delete.name} 已被删除")
```

### 代码详解

1. **查询需要删除的记录**：
   - `session.query(User).filter(User.name == "Bob").first()`：
     - 使用 `filter` 方法查找 `name` 为 `"Bob"` 的用户。
     - 使用 `first()` 方法返回第一条匹配的记录。

2. **删除记录**：
   - `session.delete(user_to_delete)`：从会话中删除该记录。

3. **提交事务**：
   - `session.commit()`：提交事务，将删除操作保存到数据库。

4. **控制台输出**：
   - 删除数据的 SQL 语句：
     ```sql
     DELETE FROM user WHERE user.id = ?
     ```

---

## 5. 增删改查的完整案例

以下是一个完整的案例，展示如何对表进行增、删、改、查的操作。

### 示例代码

```python
from sqlmodel import SQLModel, Field, create_engine, Session

# 初始化数据库引擎
engine = create_engine("sqlite:///example.db", echo=True)

# 定义 User 表
class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    name: str
    age: int

# 创建表
SQLModel.metadata.create_all(engine)

# 增：插入数据
with Session(engine) as session:
    user1 = User(name="Alice", age=25)
    user2 = User(name="Bob", age=30)
    session.add(user1)
    session.add(user2)
    session.commit()
    print("数据已插入！")

# 查：查询数据
with Session(engine) as session:
    users = session.query(User).all()
    print("查询到的用户：")
    for user in users:
        print(f"ID: {user.id}, Name: {user.name}, Age: {user.age}")

# 改：更新数据
with Session(engine) as session:
    user_to_update = session.query(User).filter(User.name == "Alice").first()
    if user_to_update:
        user_to_update.age = 26
        session.add(user_to_update)
        session.commit()
        print(f"用户 {user_to_update.name} 的年龄已更新为 {user_to_update.age}")

# 删：删除数据
with Session(engine) as session:
    user_to_delete = session.query(User).filter(User.name == "Bob").first()
    if user_to_delete:
        session.delete(user_to_delete)
        session.commit()
        print(f"用户 {user_to_delete.name} 已被删除")
```

---

## 6. 小结

- **插入数据**：使用 `session.add()` 添加记录，`session.commit()` 提交事务。
- **查询数据**：使用 `session.query()` 查询所有数据或条件查询。
- **更新数据**：查询出需要修改的记录，修改字段值后提交事务。
- **删除数据**：查询出需要删除的记录，使用 `session.delete()` 删除并提交事务。


---

# SQLModel - 一对多与多对多关系及连接表

在本节中，我们将详细讲解如何在 SQLModel 中定义和操作表之间的 **一对多** 和 **多对多** 关系。SQLModel 基于 SQLAlchemy，因此可以轻松实现复杂的表关系。以下内容包括定义关系、插入数据和查询数据的完整示例。

---

## 1. 一对多关系 (One-to-Many)

### 场景说明

在一对多关系中，一个表的某条记录可以关联到另一个表中的多条记录。例如，一个用户可以拥有多个订单。

### 示例代码

```python
from sqlmodel import SQLModel, Field, Relationship, create_engine, Session
from typing import List, Optional

# 定义数据库引擎
engine = create_engine("sqlite:///example.db", echo=True)

# 定义 User 表
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)  # 主键
    name: str                                                # 用户名
    orders: List["Order"] = Relationship(back_populates="user")  # 关联到 Order 表

# 定义 Order 表
class Order(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)  # 主键
    description: str                                           # 订单描述
    user_id: int = Field(foreign_key="user.id")                # 外键，关联到 User 表
    user: Optional[User] = Relationship(back_populates="orders")  # 关联到 User 表
```

### 代码详解

1. **`Relationship`**：
   - `Relationship` 用于定义表之间的关系。
   - 在 `User` 表中，`orders` 是一个列表，表示一个用户可以有多个订单。
   - 在 `Order` 表中，`user` 是一个对象，表示一个订单属于某个用户。

2. **`foreign_key`**：
   - 在 `Order` 表中，`user_id` 是一个外键，指向 `User` 表的 `id` 字段。

3. **`back_populates`**：
   - 用于双向绑定两个表之间的关系。
   - `User.orders` 和 `Order.user` 通过 `back_populates` 相互关联。

---

### 插入数据

```python
# 创建表
SQLModel.metadata.create_all(engine)

# 插入数据
with Session(engine) as session:
    # 创建用户
    user = User(name="Alice")
    session.add(user)
    session.commit()

    # 创建订单
    order1 = Order(description="Order 1", user_id=user.id)
    order2 = Order(description="Order 2", user_id=user.id)
    session.add(order1)
    session.add(order2)
    session.commit()

    print("数据已插入！")
```

### 查询数据

```python
# 查询数据
with Session(engine) as session:
    # 查询用户及其订单
    user = session.query(User).filter(User.name == "Alice").first()
    if user:
        print(f"用户: {user.name}")
        for order in user.orders:
            print(f"订单: {order.description}")
```

### 控制台输出

1. 插入数据时的 SQL 语句：
   ```sql
   INSERT INTO user (name) VALUES (?)
   INSERT INTO order (description, user_id) VALUES (?, ?)
   ```

2. 查询数据时的 SQL 语句：
   ```sql
   SELECT user.id, user.name FROM user WHERE user.name = ?
   SELECT order.id, order.description, order.user_id FROM order WHERE order.user_id = ?
   ```

---

## 2. 多对多关系 (Many-to-Many)

### 场景说明

在多对多关系中，一个表的某条记录可以关联到另一个表中的多条记录，反之亦然。例如，一个学生可以选修多门课程，而一门课程可以被多个学生选修。

### 示例代码

```python
from sqlmodel import SQLModel, Field, Relationship
from typing import List, Optional

# 定义连接表 StudentCourse
class StudentCourse(SQLModel, table=True):
    student_id: Optional[int] = Field(default=None, foreign_key="student.id", primary_key=True)
    course_id: Optional[int] = Field(default=None, foreign_key="course.id", primary_key=True)

# 定义 Student 表
class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    courses: List["Course"] = Relationship(back_populates="students", link_model=StudentCourse)

# 定义 Course 表
class Course(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    students: List[Student] = Relationship(back_populates="courses", link_model=StudentCourse)
```

### 代码详解

1. **连接表**：
   - `StudentCourse` 是一个连接表，用于表示学生和课程之间的多对多关系。
   - 它包含两个外键字段：`student_id` 和 `course_id`，分别指向 `Student` 和 `Course` 表的主键。

2. **`link_model`**：
   - 在 `Student` 和 `Course` 表的 `Relationship` 中，使用 `link_model` 指定连接表 `StudentCourse`。

3. **`back_populates`**：
   - `Student.courses` 和 `Course.students` 通过 `back_populates` 双向关联。

---

### 插入数据

```python
# 创建表
SQLModel.metadata.create_all(engine)

# 插入数据
with Session(engine) as session:
    # 创建学生
    student1 = Student(name="Alice")
    student2 = Student(name="Bob")

    # 创建课程
    course1 = Course(title="Math")
    course2 = Course(title="Science")

    # 建立关系
    student1.courses.append(course1)
    student1.courses.append(course2)
    student2.courses.append(course1)

    # 添加数据到会话
    session.add(student1)
    session.add(student2)
    session.commit()

    print("数据已插入！")
```

### 查询数据

```python
# 查询数据
with Session(engine) as session:
    # 查询学生及其课程
    student = session.query(Student).filter(Student.name == "Alice").first()
    if student:
        print(f"学生: {student.name}")
        for course in student.courses:
            print(f"课程: {course.title}")

    # 查询课程及其学生
    course = session.query(Course).filter(Course.title == "Math").first()
    if course:
        print(f"\n课程: {course.title}")
        for student in course.students:
            print(f"学生: {student.name}")
```

### 控制台输出

1. 插入数据时的 SQL 语句：
   ```sql
   INSERT INTO student (name) VALUES (?)
   INSERT INTO course (title) VALUES (?)
   INSERT INTO studentcourse (student_id, course_id) VALUES (?, ?)
   ```

2. 查询数据时的 SQL 语句：
   ```sql
   SELECT student.id, student.name FROM student WHERE student.name = ?
   SELECT course.id, course.title FROM course JOIN studentcourse ON course.id = studentcourse.course_id WHERE studentcourse.student_id = ?
   ```

---

## 3. 一对多与多对多的完整案例

以下是一个完整的案例，展示如何定义一对多和多对多关系，插入数据并进行查询。

### 示例代码

```python
from sqlmodel import SQLModel, Field, Relationship, create_engine, Session
from typing import List, Optional

# 定义数据库引擎
engine = create_engine("sqlite:///example.db", echo=True)

# 定义表结构
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    orders: List["Order"] = Relationship(back_populates="user")

class Order(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    description: str
    user_id: int = Field(foreign_key="user.id")
    user: Optional[User] = Relationship(back_populates="orders")

class StudentCourse(SQLModel, table=True):
    student_id: Optional[int] = Field(default=None, foreign_key="student.id", primary_key=True)
    course_id: Optional[int] = Field(default=None, foreign_key="course.id", primary_key=True)

class Student(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    courses: List["Course"] = Relationship(back_populates="students", link_model=StudentCourse)

class Course(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    students: List[Student] = Relationship(back_populates="courses", link_model=StudentCourse)

# 创建表
SQLModel.metadata.create_all(engine)

# 插入数据
with Session(engine) as session:
    # 一对多
    user = User(name="Alice")
    session.add(user)
    session.commit()

    order1 = Order(description="Order 1", user_id=user.id)
    order2 = Order(description="Order 2", user_id=user.id)
    session.add(order1)
    session.add(order2)
    session.commit()

    # 多对多
    student = Student(name="Alice")
    course = Course(title="Math")
    student.courses.append(course)
    session.add(student)
    session.commit()

# 查询数据
with Session(engine) as session:
    user = session.query(User).first()
    print(f"用户: {user.name}")
    for order in user.orders:
        print(f"订单: {order.description}")

    student = session.query(Student).first()
    print(f"\n学生: {student.name}")
    for course in student.courses:
        print(f"课程: {course.title}")
```

---

## 4. 小结

- **一对多**：通过 `Relationship` 和 `foreign_key` 建立一对多关系。
- **多对多**：通过连接表和 `link_model` 建立多对多关系。
- **插入数据**：可以通过直接操作关系字段（如 `append`）来建立关联。
- **查询数据**：通过双向关系字段（如 `back_populates`）轻松访问关联数据。


---

# 参考资料

- [SQLModel 官方文档](https://sqlmodel.tiangolo.com)

<!-- English Content -->

Content in production......

