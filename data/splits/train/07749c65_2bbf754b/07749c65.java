import java.io.*;
import java.util.*;


public class Nov1P4 {
    public static void main(String[] args) {
        MyScanner sc = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));
        long o = sc.nextInt();
        for(long i = 0; i < o; i++)
        {
            int n = sc.nextInt();
            long[] m = new long[n];
            for(int j = 0; j < n; j++)
            {
                m[j] = sc.nextLong();
            }
            String str = sc.nextLine();
            ArrayList<Long> a = new ArrayList<Long>();
            ArrayList<Long> b = new ArrayList<Long>();

            for(int j = 0; j < n; j++)
            {
                if(str.charAt(j) == 'B')
                {
                    b.add(m[j]);
                }
                else
                {
                    a.add(m[j]);
                }
            }
            Collections.sort(a);
            Collections.sort(b);
            boolean tf = true;
            for(int j = 0; j < b.size(); j++)
            {
                if(b.get(j) < (j+1) && tf)
                {
                    out.println("NO");
                    tf = false;
                }
            }
            for(int j = 0; j < a.size(); j++)
            {
                if(a.get(j) > j+1+b.size() && tf)
                {
                    out.println("NO");
                    tf = false;
                }
            }
            if(tf)
                out.println("YES");
        }
        // Start writing your solution here. -------------------------------------

      /*
      int n      = sc.nextInt();        // read input as integer
      long k     = sc.nextLong();       // read input as long
      double d   = sc.nextDouble();     // read input as double
      String str = sc.next();           // read input as String
      String s   = sc.nextLine();       // read whole line as String

      int result = 3*n;
      out.println(result);                    // print via PrintWriter
      */

        // Stop writing your solution here. -------------------------------------
        out.close();
    }


    //-----------PrintWriter for faster output---------------------------------
    public static PrintWriter out;

    //-----------MyScanner class for faster input----------
    
    //--------------------------------------------------------
}
