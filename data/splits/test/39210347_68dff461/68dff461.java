import java.io.*;
import java.util.*;
public class Main {
    public static void main(String[] args) {
        FastReader input=new FastReader();
        PrintWriter out=new PrintWriter(System.out);
        int T=input.nextInt();
        while(T-->0)
        {
            int n=input.nextInt();
            int a[]=new int[n];
            for(int i=0;i<n;i++)
            {
                a[i]=input.nextInt();
            }
            int b[]=new int[n];
            if(n%2==0)
            {
                for(int i=0;i<n;i+=2)
                {
                    int v1=Math.abs(a[i]),v2=Math.abs(a[i+1]);
                    int l=lcm(v1,v2);
                    int x=l/v1,y=l/v2;
                    if((a[i]>=0 && a[i+1]<0) || (a[i]<0 && a[i+1]>=0))
                    {
                        b[i]=x;
                        b[i+1]=y;
                    }
                    else
                    {
                        b[i]=x;
                        b[i+1]=-y;
                    }
                }
                for(int i=0;i<n;i++)
                {
                    out.print(b[i]+" ");
                }
                out.println();
            }
            else
            {
                for(int i=0;i<n-3;i+=2)
                {
                    int v1=Math.abs(a[i]),v2=Math.abs(a[i+1]);
                    int l=lcm(v1,v2);
                    int x=l/v1,y=l/v2;
                    if((a[i]>=0 && a[i+1]<0) || (a[i]<0 && a[i+1]>=0))
                    {
                        b[i]=x;
                        b[i+1]=y;
                    }
                    else
                    {
                        b[i]=x;
                        b[i+1]=-y;
                    }
                }
                int l1=lcm(Math.abs(a[n-3]),Math.abs(a[n-2]));
                int x=l1/Math.abs(a[n-3]),y=l1/Math.abs(a[n-2]);
                int l2=lcm(Math.abs(a[n-3]),Math.abs(a[n-1]));
                int z=l2/Math.abs(a[n-3]);
                if((a[n-3]>=0 && a[n-2]<0) || (a[n-3]<0 && a[n-2]>=0))
                {

                }
                else
                {
                    y=-y;
                }
                x+=z;
                int sum=a[n-3]*x+a[n-2]*y;
                sum=-sum;
                z=sum/a[n-1];
                b[n-3]=x;
                b[n-2]=y;
                b[n-1]=z;
                for(int i=0;i<n;i++)
                {
                    out.print(b[i]+" ");
                }
                out.println();
            }
        }
        out.close();
    }
    public static int gcd(int a, int b)
    {
        if(a==0)
        {
            return b;
        }
        else
        {
            return gcd(b%a,a);
        }
    }
    public static int lcm(int a, int b)
    {
        return (a/gcd(a,b))*b;
    }
    
}
