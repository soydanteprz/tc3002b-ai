import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.InputMismatchException;

public class AirConditioners {
    static InputReader inputReader=new InputReader(System.in);
    static void solve()
    {
        int n=inputReader.nextInt();
        int k=inputReader.nextInt();
        int pos[]=inputReader.nextIntArray(k);
        int power[]=inputReader.nextIntArray(k);
        int answer[]=new int[n];
        Arrays.fill(answer,(int)(2e9));
        for (int i=0;i<k;i++)
        {
            answer[pos[i]-1]=power[i];
        }
        for (int i=1;i<n;i++)
        {
            answer[i]=Math.min(answer[i],answer[i-1]+1);
        }
        for (int i=n-2;i>=0;i--)
        {
            answer[i]=Math.min(answer[i],answer[i+1]+1);
        }
        for (int i=0;i<n;i++)
        {
            out.print(answer[i]+" ");
        }
    }
    static PrintWriter out=new PrintWriter((System.out));
    public static void main(String args[])throws IOException
    {
        int t=inputReader.nextInt();
        while(t-->0)
        {
            solve();
            out.println();
        }
        out.close();
    }
    
}
