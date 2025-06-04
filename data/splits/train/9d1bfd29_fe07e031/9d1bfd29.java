import java.io.IOException;
    import java.io.InputStream;
    import java.io.PrintWriter;
    import java.util.*;
 
    public class SetorDecrease {
        static InputReader inputReader=new InputReader(System.in);
        static void solve()
        {
            int n=inputReader.nextInt();
            long k=inputReader.nextLong();
            long sum=0;
            long arr[]=new long[n];
            for (int i=0;i<n;i++)
            {
                arr[i]=inputReader.nextLong();
            }
            SortDec(arr);
            for (int i=0;i<n-1;i++)
            {
                sum+=arr[i];
            }
            long ans=Long.MAX_VALUE;
            for (int i=0;i<n;i++)
            {
               long remval=k-sum;
               long val=fdiv(remval,i+1);
               ans=Math.min(ans,i+Math.max(arr[n-1]-val,0));
               sum-=arr[i];
            }
            out.println(ans);
        }
        static void SortDec(long arr[])
        {
            List<Long>list=new ArrayList<>();
            for(long ele:arr)
            {
                list.add(ele);
            }
            Collections.sort(list,Collections.reverseOrder());
            int n=arr.length;
            for (int i=0;i<n;i++)
            {
                arr[i]=list.get(i);
 
            }
 
        }
       static long fdiv(long X, long Y) {
            if (X >= 0) {
                return X / Y;
            } else {
                return (X-Y+1) / Y;
            }
        }
        static PrintWriter out=new PrintWriter((System.out));
        public static void main(String args[])throws IOException
        {
            int t=inputReader.nextInt();
            while (t-->0) {
                solve();
            }
                out.close();
        }
        
    }
