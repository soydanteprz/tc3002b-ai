import java.io.*;
import java.util.*;
public class Main {
    public static void main(String args[])
    {
        FastReader input=new FastReader();
        PrintWriter out=new PrintWriter(System.out);
        int T=input.nextInt();
        while(T-->0)
        {
            int n=input.nextInt();
            int c[]=new int[n];
            for(int i=0;i<n;i++)
            {
                c[i]=input.nextInt();
            }
            long me=Integer.MAX_VALUE;
            long mo=Integer.MAX_VALUE;
            long se=0,so=0;
            long min=Long.MAX_VALUE;
            for(int i=1;i<=n;i++)
            {
                if(i%2==0)
                {
                    me=Math.min(me,c[i-1]);
                    se+=c[i-1];
                }
                else
                {
                    mo=Math.min(mo,c[i-1]);
                    so+=c[i-1];
                }
                if(i>=2)
                {
                    long sum=0;
                    long c1=i/2+i%2;
                    long c2=(i-1)/2+(i-1)%2;
                    if(i%2==0)
                    {
                        sum+=se;
                        sum+=(me)*(n-c1);
                        sum+=so;
                        sum+=mo*(n-c2);
                    }
                    else
                    {
                        sum+=so;
                        sum+=(mo)*(n-c1);
                        sum+=se;
                        sum+=me*(n-c2);
                    }
                    min=Math.min(min,sum);
                }
            }
            out.println(min);
        }
        out.close();
    }
    
}
