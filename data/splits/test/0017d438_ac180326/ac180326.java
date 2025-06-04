import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Collections;
import java.util.Comparator;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Stack;
import java.util.Queue;
import java.util.PriorityQueue;
import java.util.List;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.TreeSet;
import java.util.Map;
import java.util.HashMap;
import java.util.Scanner;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.Vector;

public class Dobby {

	public static void main(String[] args) {

		try {

			FastReader fr = new FastReader();
			PrintWriter pt = new PrintWriter(System.out);

			int t = fr.nextInt();

			while (t > 0) {

				int n = fr.nextInt(), m = fr.nextInt(), x = fr.nextInt();

				ArrayList<Pair> pp = new ArrayList<>();

				int A[] = new int[n];

				for (int i = 0; i < n; i++) {
					A[i] = fr.nextInt();
					Pair pr = new Pair(A[i], i);
					pp.add(pr);
				}

				Collections.sort(pp);
				Collections.reverse(pp);

				int ps[] = new int[n];
				int pk[] = new int[n];
				Arrays.fill(ps, 0);
				Arrays.fill(pk, 0);

				int index = 0;

				for (int i = 0; i < n; i++) {

					if (pk[index] < x) {
						pk[index] += pp.get(i).a;
					}
					ps[pp.get(i).b] = index + 1;
					index++;

					index = index == m ? 0 : index;

				}

				pt.println("YES");

				for (int i = 0; i < n; i++) {
					pt.print(ps[i] + " ");
				}
				pt.println();

				t--;
			}

			pt.close();

		} catch (

		Exception e) {
			return;
		}
	}

	static void merge(long arr[], int l, int m, int r) {

		int n1 = m - l + 1;
		int n2 = r - m;

		long L[] = new long[n1];
		long R[] = new long[n2];

		for (int i = 0; i < n1; ++i)
			L[i] = arr[l + i];
		for (int j = 0; j < n2; ++j)
			R[j] = arr[m + 1 + j];

		int i = 0, j = 0;

		int k = l;
		while (i < n1 && j < n2) {
			if (L[i] <= R[j]) {
				arr[k] = L[i];
				i++;
			} else {
				arr[k] = R[j];
				j++;
			}
			k++;
		}

		while (i < n1) {
			arr[k] = L[i];
			i++;
			k++;
		}

		while (j < n2) {
			arr[k] = R[j];
			j++;
			k++;
		}
	}

	static void sort(long arr[], int l, int r) {
		if (l < r) {

			int m = l + (r - l) / 2;

			sort(arr, l, m);
			sort(arr, m + 1, r);

			merge(arr, l, m, r);
		}
	}

	static class Pair implements Comparable<Pair> {
		int a, b;

		Pair(int a, int b) {
			this.a = a;
			this.b = b;
		}

		

		

	}

	

	
}
