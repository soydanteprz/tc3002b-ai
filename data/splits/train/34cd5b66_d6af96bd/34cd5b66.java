import java.io.*;
import java.util.*;

public class C {
    

    static FastReader s = new FastReader();
    static PrintWriter out = new PrintWriter(System.out);

    

    

    

    

    private static int ri() {
        return s.nextInt();
    }

    

    private static String rs() {
        return s.next();
    }

    

    
    static int MOD=1000000007;



    



    public static void main(String[] args) {
        StringBuilder ans = new StringBuilder();
        int t = ri();
//        int t = 1;
        while (t-- > 0)
        {
            int n=ri();
            int m=ri();
            int[][] arr=new int[n][m];
            for(int i=0;i<n;i++)
            {
                char[] mid= rs().toCharArray();
                for(int j=0;j<m;j++)
                {
                    arr[i][j]=mid[j]-'0';
                }
            }

            int res=0;
            StringBuilder str=new StringBuilder();
            for(int i=0;i<n;)
            {
                for(int j=0;j<m;)
                {
                    boolean[] vis=new boolean[4];
                    HashSet<Point> set=new HashSet<>();
                    int count=0;
                    if(arr[i][j]==1)
                    {
                        count++;
                        set.add(new Point(i,j));
                        vis[0]=true;
                    }
                    if(arr[i][j+1]==1)
                    {
                        count++;
                        set.add(new Point(i,j+1));
                        vis[1]=true;
                    }
                    if(arr[i+1][j]==1)
                    {
                        count++;
                        set.add(new Point(i+1,j));
                        vis[2]=true;
                    }
                    if(arr[i+1][j+1]==1)
                    {
                        count++;
                        set.add(new Point(i+1,j+1));
                        vis[3]=true;
                    }

                    if(count==4)
                    {
                        int v1=i+1;
                        int v2=j+1;
                        res++;
                        count=1;
                        str.append(v1).append(" ").append(v2).append(" ").append(v1).append(" ").append(v2 + 1).append(" ").append(v1 + 1).append(" ").append(v2).append("\n");
                        vis[0]=false;
                        vis[1]=false;
                        vis[2]=false;
                        arr[i][j]=0;
                        arr[i][j+1]=0;
                        arr[i+1][j]=0;
                    }
                    if(count==1)
                    {
                        res++;
                        count=2;
                        int p1=i+1,p2=j+1;
                        if(vis[0])
                        {
                            str.append(p1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2+1).append(" ").append(p1 + 1).append(" ").append(p2).append("\n");
                            vis[0]=false;
                            vis[1]=true;
                            vis[2]=true;
                            arr[i][j]=1;
                            arr[i][j+1]=0;
                            arr[i+1][j]=0;

                        }
                        else if(vis[1])
                        {
                            p2++;
                            str.append(p1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2-1).append(" ").append(p1 + 1).append(" ").append(p2).append("\n");
                            vis[1]=false;
                            vis[0]=true;
                            vis[3]=true;
                            arr[i][j+1]=1;
                            arr[i][j]=0;
                            arr[i+1][j+1]=0;
                        }
                        else if(vis[2])
                        {
                            p1++;
                            str.append(p1).append(" ").append(p2).append(" ").append(p1-1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2+1).append("\n");
                            vis[2]=false;
                            vis[0]=true;
                            vis[3]=true;
                            arr[i+1][j]=1;
                            arr[i][j]=0;
                            arr[i+1][j+1]=0;
                        }
                        else {
                            p1++;
                            p2++;
                            str.append(p1).append(" ").append(p2).append(" ").append(p1-1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2-1).append("\n");
                            vis[3]=false;
                            vis[1]=true;
                            vis[2]=true;
                            arr[i+1][j+1]=1;
                            arr[i][j+1]=0;
                            arr[i+1][j]=0;
                        }
                    }
                    if(count==2)
                    {
                        res++;
                        count=3;
                        int p1=i+1,p2=j+1;

                        if(vis[0]&&vis[1])
                        {
                            str.append(p1).append(" ").append(p2).append(" ").append(p1+1).append(" ").append(p2).append(" ").append(p1+1).append(" ").append(p2+1).append("\n");
                            vis[0]=false;
                            vis[2]=true;
                            vis[3]=true;
                            arr[i][j]=1;
                            arr[i+1][j]=0;
                            arr[i+1][j+1]=0;
                        }
                        else if(vis[0] && vis[2])
                        {
                            str.append(p1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2+1).append(" ").append(p1+1).append(" ").append(p2+1).append("\n");
                            vis[0]=false;
                            vis[1]=true;
                            vis[3]=true;
                            arr[i][j]=1;
                            arr[i][j+1]=0;
                            arr[i+1][j+1]=0;
                        }
                        else if(vis[0] && vis[3])
                        {
                            str.append(p1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2+1).append(" ").append(p1+1).append(" ").append(p2).append("\n");
                            vis[0]=false;
                            vis[1]=true;
                            vis[2]=true;
                            arr[i][j]=1;
                            arr[i][j+1]=0;
                            arr[i+1][j]=0;
                        }
                        else if(vis[1] && vis[2])
                        {
                            str.append(p1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2+1).append(" ").append(p1+1).append(" ").append(p2+1).append("\n");
                            vis[1]=false;
                            vis[0]=true;
                            vis[3]=true;
                            arr[i][j+1]=1;
                            arr[i][j]=0;
                            arr[i+1][j+1]=0;
                        }
                        else if(vis[1] && vis[3])
                        {
//                            System.out.println("yoyo");
                            str.append(p1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2+1).append(" ").append(p1+1).append(" ").append(p2).append("\n");
                            vis[1]=false;
                            vis[0]=true;
                            vis[2]=true;
                            arr[i][j+1]=1;
                            arr[i][j]=0;
                            arr[i+1][j]=0;
                        }
                        else if(vis[2] && vis[3])
                        {
                            str.append(p1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2+1).append(" ").append(p1+1).append(" ").append(p2).append("\n");
                            vis[2]=false;
                            vis[0]=true;
                            vis[1]=true;
                            arr[i+1][j]=1;
                            arr[i][j]=0;
                            arr[i][j+1]=0;
                        }
                    }
                    if(count==3)
                    {
                        res+=1;
                        int p1=i+1,p2=j+1;
                        if(vis[0] && vis[1] && vis[2])
                        {
                            str.append(p1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2+1).append(" ").append(p1+1).append(" ").append(p2).append("\n");

                        }
                        else if(vis[0] && vis[1] && vis[3])
                        {
                            str.append(p1).append(" ").append(p2).append(" ").append(p1).append(" ").append(p2+1).append(" ").append(p1+1).append(" ").append(p2+1).append("\n");

                        }
                        else if(vis[0] && vis[2] && vis[3])
                        {
                            str.append(p1).append(" ").append(p2).append(" ").append(p1+1).append(" ").append(p2+1).append(" ").append(p1+1).append(" ").append(p2).append("\n");

                        }
                        else if(vis[3] && vis[1] && vis[2])
                        {
                            str.append(p1+1).append(" ").append(p2+1).append(" ").append(p1).append(" ").append(p2+1).append(" ").append(p1+1).append(" ").append(p2).append("\n");

                        }
                        arr[i][j]=0;
                        arr[i][j+1]=0;
                        arr[i+1][j]=0;
                        arr[i+1][j+1]=0;


                    }




                    j+=2;
                    if(j==m-1)
                    {
                       j--;
                    }
                }
                i+=2;
                if(i==n-1)
                {
                    i--;
                }
            }
            ans.append(res).append("\n");
            ans.append(str.toString());

        }


        out.print(ans.toString());
        out.flush();

    }
    static class Point
    {
        int x,y;

        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

}
