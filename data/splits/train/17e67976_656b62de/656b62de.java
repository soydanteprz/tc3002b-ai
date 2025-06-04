import java.io.IOException;
    import java.io.InputStream;
    import java.io.PrintWriter;
    import java.util.InputMismatchException;
    import java.util.PriorityQueue;

    public class InterestingStory {
        static InputReader inputReader=new InputReader(System.in);
        static void solve()
        {
            int n=inputReader.nextInt();
            String arr[]=new String[n];
            for (int i=0;i<n;i++)
            {
                arr[i]=inputReader.readString();
            }
            int larr[]=new int[n];
            for (int i=0;i<n;i++) {
                larr[i] = arr[i].length();
            }
            int freqarr[][]=new int[n][5];
            for (int i=0;i<n;i++)
            {
                freqarr[i]=freq(arr[i],larr[i]);
            }
            int max=Integer.MIN_VALUE;
            for (char c='a';c<='e';c++)
            {
                max=Math.max(max,helper(freqarr,larr,n,c));
            }
            out.println(max);
        }
        static  int helper(int freqarr[][],int larr[],int n,char c)
        {
            int sum=0;
            PriorityQueue<Integer>priorityQueue=new PriorityQueue<>();
            for (int i=0;i<n;i++)
            {
                int score=freqarr[i][c-'a']-(larr[i]-freqarr[i][c-'a']);
                priorityQueue.add(score);
                sum+=score;
            }
            while (!priorityQueue.isEmpty()&&sum<=0) {
                sum -= priorityQueue.poll();
            }
            return priorityQueue.size();
        }
       static int []freq(String str,int len)
        {
            int arr[]=new int[5];
            for (int i=0;i<len;i++)
            {
                arr[str.charAt(i)-'a']++;
            }
            return arr;
        }
        static PrintWriter out=new PrintWriter((System.out));
        public static void main(String args[])throws IOException
        {
            int t=inputReader.nextInt();
            while(t-->0)
            {
                solve();
            }
            out.close();
        }
        
    }
