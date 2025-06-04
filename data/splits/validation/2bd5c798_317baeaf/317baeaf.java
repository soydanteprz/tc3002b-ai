import java.util.*;
import java.io.*;

public class Main {
  public static void main(String[] args) throws Exception {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		PrintWriter pw = new PrintWriter(System.out);
		StringTokenizer st;

		int t = Integer.parseInt(br.readLine());
		while (t --> 0) {
			int n = Integer.parseInt(br.readLine());
			String a = br.readLine();
			String b = br.readLine();
			int alit = 0;
			int blit = 0;

			int ans = Integer.MAX_VALUE;

			for (int i = 0; i < n; i++) {
				if (a.charAt(i) == '1') alit++;
				if (b.charAt(i) == '1') blit++;
			}

			if (alit == blit) {
				int count = 0;
				for (int i = 0; i < n; i++)
					if (a.charAt(i) != b.charAt(i)) count++;
				ans = Math.min(count, ans);
			}
			
			if (alit == n - blit + 1) {
				int count = 0;
				for (int i = 0; i < n; i++)
					if (a.charAt(i) == b.charAt(i)) count++;
				ans = Math.min(ans, count);
			}

			if (ans == Integer.MAX_VALUE) {
				pw.println("-1");
			} else {
				pw.println(ans);
			}
		}

		pw.close();
  }
}
