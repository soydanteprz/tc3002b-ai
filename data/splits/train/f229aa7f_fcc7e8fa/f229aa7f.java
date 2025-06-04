import java.util.*;
import java.io.*;
import java.math.*;

public class cf {
	static PrintWriter pw = new PrintWriter(System.out);

	public static void main(String[] args) throws IOException, InterruptedException {
		Scanner sc = new Scanner(System.in);
		int t = sc.nextInt();
		while (t-- > 0) {
			int n = sc.nextInt();
			char[] a = sc.next().toCharArray();
			char[] b = sc.next().toCharArray();
			int x = 0, y = 0, lit = 0,lit2 = 0;
			for (int i = 0; i < n; i++) {
				if (a[i] == '1')
					lit++;
				if (b[i] == '1')
					lit2++;
				if (a[i] == b[i])
					x++;
				else
					y++;
			}
			if(lit == lit2 || n - lit + 1 == lit2) {
				if (lit == lit2 && n - lit + 1 == lit2) {
					pw.println(Math.min(x,y));
				}else if(lit == lit2) {
					pw.println(y);
				}else {
					pw.println(x);
				}
			}else {
				pw.println(-1);
			}
		}
		pw.close();
	}

	

	

	

}
