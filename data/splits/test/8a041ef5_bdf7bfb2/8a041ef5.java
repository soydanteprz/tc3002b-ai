import java.util.*;
import java.io.*;
import java.lang.*;
import java.util.jar.JarEntry;

public class Main{
    /* || श्री गणेशाय नमः ||
       @𝐚𝐮𝐭𝐡𝐨𝐫 𝐉𝐢𝐠𝐚𝐫 𝐍𝐚𝐢𝐧𝐮𝐣𝐢
       𝐒𝐕𝐍𝐈𝐓- 𝐒𝐮𝐫𝐚𝐭
    */
    public static void main(String args[]){
        InputReader in=new InputReader(System.in);
        TASK solver = new TASK();
        int t=1;
        t = in.nextInt();
        for(int i=1;i<=t;i++)
        {
            solver.solve(in,i);
        }
    }
    static class TASK {


        void solve(InputReader in, int testNumber) {
           int n = in.nextInt();
           int a[][] = new int[n][5];
           for(int i=0;i<n;i++)
           {
               char[] s = in.next().toCharArray();
               for(int j=0;j<s.length;j++)
               {
                   a[i][s[j]-'a']++;
               }
           }
           int max=0;
           int x[] = new int[n];
           for(int j=0;j<=4;j++)
           {
               for(int i=0;i<n;i++)
               {
                   x[i]=2*a[i][j];
                   for(int k=0;k<5;k++)
                   {
                       x[i]-=a[i][k];
                   }
               }
               Arrays.sort(x);
               int c=0,sum=0;
               for(int i=n-1;i>=0;i--)
               {
                   sum+=x[i];
                   if(sum<=0)
                       break;
                   c++;
               }
               max=Math.max(max,c);
           }
            System.out.println(max);
















        }
    }


    

    
    
}
