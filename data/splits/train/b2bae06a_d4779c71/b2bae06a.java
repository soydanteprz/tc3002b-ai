import java.util.*;
    import java.io.*;
    public class D{
        
        
         
        static int mod = (int)(1e9+7);
        

        public static void main(String[] args) {
            InputReader in = new InputReader(System.in);
            OutputWriter out = new OutputWriter(System.out);
            int n = in.nextInt();
            int[] arr = in.nextIntArray(n);
            Stack<Integer> min = new Stack<>();
            Stack<Integer> max = new Stack<>();
            int[] dp = new int[n];
            // Arrays.fill(dp,(int)1e9);
            dp[0] = 0;
            min.push(0);
            max.push(0);
            for(int i=1;i<n;i++)
            {
                int h=dp[i-1]+1;
                while(!max.isEmpty() && arr[i]>arr[max.peek()])
                {
                    int x = arr[max.peek()];
                    h = Math.min(h,1+dp[max.pop()]);
                    while(!max.isEmpty() && arr[max.peek()]==x)
                    {
                        max.pop();
                    }
                }
                if(!max.isEmpty())
                {
                    h = Math.min(h,1+dp[max.peek()]);
                }
                while(!min.isEmpty() && arr[i]<arr[min.peek()])
                {
                    int x = arr[min.peek()];
                    h = Math.min(h,1+dp[min.pop()]);
                    while(!min.isEmpty() && arr[min.peek()]==x)
                    {
                        min.pop();
                    }
                }
                if(!min.isEmpty())
                {
                    h = Math.min(h,1+dp[min.peek()]);
                }
                dp[i] = h;
                min.push(i);
                max.push(i);
            }
            out.printLine(dp[n-1]);
            out.flush();
            out.close();
        }
    }
