#功能：将图片导入到MySQL数据库
import pymysql
import datetime

def checkAccount(acccout_name,password):
    conn = pymysql.connect(host='127.0.0.1', user='root', passwd="123456", db='clock')
    cur = conn.cursor()
    # SQL 查询语句
    sql = "SELECT * FROM accountinfo WHERE Account=%s AND Password=%s"
    try:
        # 执行SQL语句
        cur.execute(sql,(acccout_name,password))
        # 获取所有记录列表
        conn.commit()
        results = cur.fetchall()

        print("results",results)
        if(len(results)>0):
            print("not none！")
            return 1
        else:
            print("none！")
            return 0
        conn.commit()
        cur.close()
        conn.close()
    except:
        print("problem!")
        return 0



#注册信息入库
def WriteRegisterIntoSQL(acccout_name,password):
    print("enter")
    conn = pymysql.connect(host='127.0.0.1', user='root', passwd="123456", db='clock')
    cur = conn.cursor()
    print(cur)
    sql = "INSERT INTO accountinfo (Account,Password) VALUES  (%s,%s)"
    cur.execute(sql, (acccout_name,password))
    conn.commit()
    cur.close()
    conn.close()
    print("============")
    print("Done! ")

#签到信息入库
def WriteIntoSQL(acccout_time,user):
    conn = pymysql.connect(host='127.0.0.1', user='root', passwd="123456", db='clock')
    cur = conn.cursor()
    sql = "INSERT INTO usersigntime (Time,User) VALUES  (%s,%s)"
    cur.execute(sql, (acccout_time,user))
    conn.commit()
    cur.close()
    conn.close()
    print("============")
    print("Done! ")

#WriteIntoSQL("2019-12-08 16:47:20","fyc")
curr_time = datetime.datetime.now()
time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')  # 2019-07-06 15:50:12
pred_name="fengyicheng"
# MYSQL部分
WriteIntoSQL(time_str, pred_name)
