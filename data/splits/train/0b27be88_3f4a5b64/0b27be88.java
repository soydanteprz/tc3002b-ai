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
    
    
    
    
    
    
    
    public static void main(String args[]) throws IOException {
        Scan input=new Scan();
        StringBuilder ans=new StringBuilder("");
        int test=input.scanInt();
        for(int tt=1;tt<=test;tt++) {
            int n=input.scanInt();
            int m=input.scanInt();
            int x=input.scanInt();
            ans.append("YES\n");
            TreeMap<Integer,Stack<Integer>> tm=new TreeMap<>();
            for(int i=1;i<=m;i++) {
                ans.append(i+" ");
                int tmp=input.scanInt();
                if(!tm.containsKey(tmp)) {
                    Stack<Integer> stck=new Stack<>();
                    stck.add(i);
                    tm.put(tmp, stck);
                }
                else {
                    Stack<Integer> stck=tm.get(tmp);
                    stck.add(i);
                }
            }
            for(int i=m;i<n;i++) {
                int tmp=input.scanInt();
                int min=tm.firstKey();
                Stack<Integer> stck=tm.get(min);
                int indx=stck.pop();
                ans.append(indx+" ");
                if(stck.isEmpty()) {
                    tm.remove(min);
                }
                tmp+=min;
                
                if(!tm.containsKey(tmp)) {
                    stck=new Stack<>();
                    stck.add(indx);
                    tm.put(tmp, stck);
                }
                else {
                    stck=tm.get(tmp);
                    stck.add(indx);
                }
            }
            ans.append("\n");
        }
        System.out.println(ans);
    }
}
