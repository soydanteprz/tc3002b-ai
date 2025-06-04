import java.util.*;
import java.io.*;
public class C {
    static class Scan {
        private byte[] buf=new byte[1024];
        private int index;
        private InputStream in;
        private int total;
        public Scan()
        {
            in=System.in;
        }
        public int scan()throws IOException
        {
            if(total<0)
            throw new InputMismatchException();
            if(index>=total)
            {
                index=0;
                total=in.read(buf);
                if(total<=0)
                return -1;
            }
            return buf[index++];
        }
        public int scanInt()throws IOException
        {
            int integer=0;
            int n=scan();
            while(isWhiteSpace(n))
            n=scan();
            int neg=1;
            if(n=='-')
            {
                neg=-1;
                n=scan();
            }
            while(!isWhiteSpace(n))
            {
                if(n>='0'&&n<='9')
                {
                    integer*=10;
                    integer+=n-'0';
                    n=scan();
                }
                else throw new InputMismatchException();
            }
            return neg*integer;
        }
        
        
        private boolean isWhiteSpace(int n)
        {
            if(n==' '||n=='\n'||n=='\r'||n=='\t'||n==-1)
            return true;
            return false;
        }
    }
    
    
    
    
    
    
    
    static int n;
    static long arr[];
    public static void main(String args[]) throws IOException {
        Scan input=new Scan();
        StringBuilder ans=new StringBuilder("");
        int test=input.scanInt();
        for(int tt=1;tt<=test;tt++) {
            n=input.scanInt();
            arr=new long[n];
            for(int i=0;i<n;i++) {
                arr[i]=input.scanInt();
            }
            ans.append(solve()+"\n");
        }
        System.out.println(ans);
    }
    
    public static long solve() {
        long ans=Long.MAX_VALUE;
        long r_min=Long.MAX_VALUE,u_min=Long.MAX_VALUE;
        long r_rem=n,u_rem=n,sum=0;
        
        for(int i=0;i<n;i++) {
            if(i%2==0) {
                r_min=Math.min(r_min,arr[i]);
                r_rem--;
            }
            else {
                u_min=Math.min(u_min,arr[i]);
                u_rem--;
            }
            sum+=arr[i];
            if(i==0) {
                continue;
            }
            
            ans=Math.min(ans,sum+(r_rem*r_min)+(u_rem*u_min));
        }
        
        r_min=Long.MAX_VALUE;
        u_min=Long.MAX_VALUE;
        r_rem=n;
        u_rem=n;
        sum=0;
        
        for(int i=0;i<n;i++) {
            if(i%2==1) {
                r_min=Math.min(r_min,arr[i]);
                r_rem--;
            }
            else {
                u_min=Math.min(u_min,arr[i]);
                u_rem--;
            }
            sum+=arr[i];
            if(i==0) {
                continue;
            }
            
            ans=Math.min(ans,sum+(r_rem*r_min)+(u_rem*u_min));
        }
        
        return ans;
    }
}
