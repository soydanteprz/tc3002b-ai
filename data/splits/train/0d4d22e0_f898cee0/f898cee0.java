import java.util.*;
import java.io.*;
public class C{
    
    
     
    static int mod = (int)(1e9+7);
    
    
    static void flip(char[][] arr,int i,int j)
    {
        if(arr[i][j]=='1')
            arr[i][j] = '0';
        else
            arr[i][j] = '1';
    }
    static void add2s(StringBuilder sb,int n,int m)
    {
        sb.append((n) +" " + (m-1) + " "  +(n-1) + " " + (m) + " "  +(n) + " " + (m)+"\n");
        sb.append((n-1) +" " + (m-1) + " "  +(n-1) + " " + (m) + " "  +(n) + " " + (m)+"\n");
    }
    static void add3s(StringBuilder sb,int n,int m)
    {
        sb.append((n) +" " + (m-1) + " "  +(n-1) + " " + (m-1) + " "  +(n) + " " + (m)+"\n");
        sb.append((n-1) +" " + (m-1) + " "  +(n) + " " + (m-1) + " "  +(n-1) + " " + (m)+"\n");
    }
    public static void main(String[] args) {
        InputReader in = new InputReader(System.in);
        OutputWriter out = new OutputWriter(System.out);
        int t = in.nextInt();
        while(t-- >0)
        {
            // out.printLine("UAShd");
            int n = in.nextInt(),m = in.nextInt();
            char[][] arr = new char[n][m];
            for(int i=0;i<n;i++)
                arr[i] = in.nextLine().toCharArray();
            // boolean[][] flip = new boolean[n][m];
            int nops = 0;
            StringBuilder res = new StringBuilder();
            //do till row n-1
            for(int i=0;i<n-1;i++)
            {
                for(int j=0;j<m;j++)
                {
                    if(arr[i][j]=='1')
                    {
                        nops++;
                        if(j==m-1)
                        {
                            res.append((i+1) + " " + (j+1) + " " + (i+2) + " " + (j+1) + " " + (i+2) + " " + (j) + "\n");
                            flip(arr,i,j);
                            flip(arr,i+1,j);
                            flip(arr,i+1,j-1);
                        }
                        else
                        {
                            res.append((i+1) + " " + (j+1) + " " + (i+2) + " " + (j+1) + " " + (i+2) + " " + (j+2) + "\n");
                            flip(arr,i,j);
                            flip(arr,i+1,j);
                            flip(arr,i+1,j+1);
                        }
                    }
                }
            }
            // out.printLine(nops);
            int i = n-2;
            for(int j=0;j<m-1;j++)
            {
                char a = arr[i][j],b = arr[i+1][j];
                if(a=='0' && b=='0')
                {
                    continue;
                }
                else if(a=='0' && b=='1')
                {
                    nops++;
                    res.append((i+2) + " " + (j+1) + " " + (i+1) + " " + (j+2) + " " + (i+2) + " " + (j+2) + "\n");
                    flip(arr,i+1,j);
                    flip(arr,i,j+1);
                    flip(arr,i+1,j+1);
                }
                else if(a=='1' && b=='0')
                {
                    nops++;
                    res.append((i+1) + " " + (j+1) + " " + (i+1) + " " + (j+2) + " " + (i+2) + " " + (j+2) + "\n");
                    flip(arr,i,j);
                    flip(arr,i,j+1);
                    flip(arr,i+1,j+1);
                }
                else
                {
                    nops++;
                    res.append((i+2) + " " + (j+1) + " " + (i+1) + " " + (j+1) + " " + (i+2) + " " + (j+2) + "\n");
                    flip(arr,i+1,j);
                    flip(arr,i,j);
                    flip(arr,i+1,j+1);
                }
            }
            // out.printLine(nops);
            char a = arr[n-2][m-1];
            char b = arr[n-1][m-1];
            if(a=='0' && b=='0')
            {

            }
            else if(a=='1' && b=='0')
            {
                nops+=3;
                res.append((n-1) + " " + (m) + " " + (n-1) + " " + (m-1) + " " + (n) + " " + (m-1)+"\n");
                add2s(res,n,m);
            }
            else if(a=='0' && b=='1')
            {
                nops+=3;
                res.append((n) + " " + (m) + " " + (n-1) + " " + (m-1) + " " + (n) + " " + (m-1)+"\n");
                add2s(res,n,m);
            }
            else
            {
                nops+=2;
                add3s(res,n,m);
            }
            // out.printLine("UAShd");
            // for(i=0;i<n;i++)
                // out.printLine(new String(arr[i]));
            // if(nops>0)
                // res.deleteCharAt(res.length()-1);
            out.printLine(nops);
            if(nops>0)
                out.print(res);
        }
        out.flush();
        out.close();
    }
}
